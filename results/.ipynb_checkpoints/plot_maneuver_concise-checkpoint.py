from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = Path("results_by_maneuver.csv")
OUT_DIR = Path("figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 读数据
df = pd.read_csv(CSV_PATH)

# 规范顺序
maneuvers = ["straight", "left_turn", "right_turn"]
horizons = [1.0, 2.0, 3.0]  # 秒
models_order = ["CV", "VanillaLSTM", "TurnAwareLSTM", "TinyTransformer"]
available_models = [m for m in models_order if m in df["model"].unique()]


def plot_metric_bigfig(metric: str, outname: str):
    """
    metric: 'ADE' or 'FDE'
    生成一张大图，包含 3 个子图：
      子图1: straight
      子图2: left_turn
      子图3: right_turn
    每个子图横轴为 horizon(1/2/3s)，每个 horizon 处画多根柱子(不同模型)。
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, man in zip(axes, maneuvers):
        sub = df[df["maneuver"] == man].copy()

        # 准备每个 horizon × model 的矩阵
        vals = np.full((len(horizons), len(available_models)), np.nan, dtype=float)
        for i, H in enumerate(horizons):
            subH = sub[sub["horizon_s"] == H]
            for j, mod in enumerate(available_models):
                sel = subH[subH["model"] == mod]
                if len(sel):
                    vals[i, j] = float(sel[metric].values[0])

        x = np.arange(len(horizons))  # 0,1,2
        width = 0.18

        for j, mod in enumerate(available_models):
            ax.bar(
                x + (j - (len(available_models) - 1) / 2) * width,
                vals[:, j],
                width=width,
                label=mod,
            )

        ax.set_title(f"{man}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(h)}s" for h in horizons])
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        if ax is axes[0]:
            ax.set_ylabel(metric)

    # 统一图例（放在外侧）
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(available_models),
        frameon=False,
        bbox_to_anchor=(0.5, 1.12),
    )
    fig.suptitle(f"{metric} by Maneuver across Horizons", y=1.08)
    fig.tight_layout()
    outfile = OUT_DIR / outname
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")


# 生成两张总图：ADE / FDE
plot_metric_bigfig("ADE", "all_in_one_ADE.png")
plot_metric_bigfig("FDE", "all_in_one_FDE.png")
