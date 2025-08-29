import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# 你的模型（保持一致）
# -----------------------------
class TrajectoryPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=2):
        super().__init__()
        self.lstm_encoder = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.lstm_decoder = nn.LSTM(
            output_size, hidden_size, num_layers, batch_first=True
        )
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, target_len):
        _, (hidden, cell) = self.lstm_encoder(x)
        decoder_input = x[:, -1, :2].unsqueeze(1)  # 只用最后一个位置作为decoder起点
        outs = []
        for _ in range(target_len):
            out, (hidden, cell) = self.lstm_decoder(decoder_input, (hidden, cell))
            out = self.fc_out(out)
            outs.append(out.squeeze(1))
            decoder_input = out
        return torch.stack(outs, dim=1)


# -----------------------------
# 工具函数
# -----------------------------
def majority_label(arr):
    if len(arr) == 0:
        return "straight"
    c = Counter(arr)
    return c.most_common(1)[0][0]


def compute_ade_fde(pred_xy, gt_xy):
    # pred_xy, gt_xy: (T, 2) 已反标准化
    diffs = pred_xy - gt_xy
    dists = np.linalg.norm(diffs, axis=1)
    ade = float(dists.mean())
    fde = float(dists[-1])
    return ade, fde


def bar_with_err(means, stds, labels, ylabel, title, out_png):
    x = np.arange(len(labels))
    plt.figure(figsize=(6, 4))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels, rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -----------------------------
# 主流程
# -----------------------------
def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读数据 & 合并
    df_trk = pd.read_csv(
        Path(args.data_dir) / "tracking_data.csv"
    )  # 需包含: id, frame, center_x, center_y
    df_turn = pd.read_csv(
        Path(args.data_dir) / "overall_turn_label.csv"
    )  # 需包含: id, frame, overall_turn_label
    df = pd.merge(
        df_trk,
        df_turn[["id", "frame", "overall_turn_label"]],
        on=["id", "frame"],
        how="left",
    )

    # 填充标签（前后向）
    df["overall_turn_label"] = df.groupby("id")["overall_turn_label"].ffill()
    df["overall_turn_label"] = df.groupby("id")["overall_turn_label"].bfill()
    df["overall_turn_label"] = df["overall_turn_label"].fillna("straight")

    # One-Hot
    enc = OneHotEncoder(sparse_output=False)
    turn_oh = enc.fit_transform(df[["overall_turn_label"]])
    turn_cols = enc.get_feature_names_out(["overall_turn_label"])
    for i, c in enumerate(turn_cols):
        df[c] = turn_oh[:, i]

    # 2) 造序列（同时记录窗口多数票标签）
    obs_len, pred_len = args.obs_len, args.pred_len
    feats = ["center_x", "center_y"] + list(turn_cols)  # 和你之前一致：2 + 3 = 5
    seq_inputs, seq_targets, seq_labels = [], [], []
    seq_vehicle_ids = []

    for vid, g in df.sort_values(["id", "frame"]).groupby("id", sort=False):
        g = g.reset_index(drop=True)
        X = g[feats].values
        lbl_seq = g["overall_turn_label"].values
        n = len(X) - obs_len - pred_len + 1
        if n <= 0:
            continue
        for i in range(n):
            x_win = X[i : i + obs_len]
            y_win = X[i + obs_len : i + obs_len + pred_len, :2]  # 预测的是真实坐标
            lbl_win = lbl_seq[i : i + obs_len]
            seq_inputs.append(x_win)
            seq_targets.append(y_win)
            seq_labels.append(majority_label(lbl_win))
            seq_vehicle_ids.append(vid)

    seq_inputs = np.asarray(seq_inputs, dtype=np.float32)  # (N, obs, F)
    seq_targets = np.asarray(seq_targets, dtype=np.float32)  # (N, pred, 2)
    seq_labels = np.asarray(seq_labels)
    seq_vehicle_ids = np.asarray(seq_vehicle_ids)

    # 3) 标准化（仅 x,y）
    xy_idx = [0, 1]
    scaler = (
        joblib.load(args.scaler)
        if args.scaler and os.path.exists(args.scaler)
        else None
    )
    if scaler is None:
        scaler = StandardScaler()
        all_xy = seq_inputs[:, :, xy_idx].reshape(-1, len(xy_idx))
        scaler.fit(all_xy)
    seq_inputs[:, :, xy_idx] = scaler.transform(
        seq_inputs[:, :, xy_idx].reshape(-1, len(xy_idx))
    ).reshape(seq_inputs.shape[0], seq_inputs.shape[1], len(xy_idx))
    seq_targets = scaler.transform(seq_targets.reshape(-1, len(xy_idx))).reshape(
        seq_targets.shape[0], seq_targets.shape[1], len(xy_idx)
    )

    # 4) 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryPredictor(
        input_size=seq_inputs.shape[2], hidden_size=128, num_layers=2, output_size=2
    ).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 5) 推理 & 计算 ADE/FDE
    rows = []
    with torch.no_grad():
        B = args.batch_size
        N = len(seq_inputs)
        for s in range(0, N, B):
            e = min(N, s + B)
            x = torch.from_numpy(seq_inputs[s:e]).to(device)
            y_true = torch.from_numpy(seq_targets[s:e]).to(device)
            y_pred = model(x, target_len=pred_len).cpu().numpy()
            y_true = y_true.cpu().numpy()

            # 反标准化
            y_pred_un = scaler.inverse_transform(y_pred.reshape(-1, 2)).reshape(
                y_pred.shape
            )
            y_true_un = scaler.inverse_transform(y_true.reshape(-1, 2)).reshape(
                y_true.shape
            )

            for i in range(y_pred_un.shape[0]):
                ade, fde = compute_ade_fde(y_pred_un[i], y_true_un[i])
                rows.append({"maneuver": seq_labels[s + i], "ade": ade, "fde": fde})

    res_df = pd.DataFrame(rows)
    res_df.to_csv(out_dir / "per_sequence_metrics.csv", index=False)

    # 6) 聚合 & 输出
    order = pd.CategoricalDtype(["left", "right", "straight"], ordered=True)
    if not set(["left", "right", "straight"]).issubset(
        set(res_df["maneuver"].unique())
    ):
        # 兼容 onehot 的列名不同（例如 overall_turn_label_left ...）
        # 这里做个简单映射：包含 'left' 的都归 left，右同理；其余归 straight
        res_df["maneuver"] = res_df["maneuver"].astype(str)
        res_df.loc[res_df["maneuver"].str.contains("left", case=False), "maneuver"] = (
            "left"
        )
        res_df.loc[res_df["maneuver"].str.contains("right", case=False), "maneuver"] = (
            "right"
        )
        res_df.loc[~res_df["maneuver"].isin(["left", "right"]), "maneuver"] = "straight"

    res_df["maneuver"] = res_df["maneuver"].astype(order)
    summary = (
        res_df.groupby("maneuver")
        .agg(
            n=("ade", "size"),
            ADE_mean=("ade", "mean"),
            ADE_std=("ade", "std"),
            FDE_mean=("fde", "mean"),
            FDE_std=("fde", "std"),
        )
        .reset_index()
        .sort_values("maneuver")
    )
    summary.to_csv(out_dir / "summary_by_maneuver.csv", index=False)
    print(summary)

    # 7) 画图（柱状图）
    labels = list(summary["maneuver"].astype(str))
    bar_with_err(
        summary["ADE_mean"].values,
        summary["ADE_std"].values,
        labels,
        "ADE (pix or m)",
        "ADE by Maneuver",
        out_dir / "ade_by_maneuver.png",
    )
    bar_with_err(
        summary["FDE_mean"].values,
        summary["FDE_std"].values,
        labels,
        "FDE (pix or m)",
        "FDE by Maneuver",
        out_dir / "fde_by_maneuver.png",
    )

    print(f"Done. Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--scaler", type=str, default="")
    ap.add_argument("--obs_len", type=int, default=90)
    ap.add_argument("--pred_len", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--out_dir", type=str, default="exper_turn_split_out")
    args = ap.parse_args()
    main(args)
