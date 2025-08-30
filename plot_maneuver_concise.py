from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------
# I/O
# -----------------------
CSV_PATH = Path("results_by_maneuver.csv")
OUT_DIR = Path("figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# -----------------------
# Settings
# -----------------------
models_order = ["CV", "VanillaLSTM", "TurnAwareLSTM", "TinyTransformer"]
maneuvers = ["straight", "left_turn", "right_turn"]
horizons = [1.0, 2.0, 3.0]  # seconds
metrics = ["ADE", "FDE"]  # two figures

# 只保留可用模型的顺序
available_models = [m for m in models_order if m in df["model"].unique()]
if not available_models:
    raise ValueError("No known models found in CSV. Check 'model' column.")


def _values_for_panel(sub_df, metric: str):
    """
    返回 shape=(len(maneuvers), len(available_models)) 的数值矩阵；
    行=man, 列=model
    """
    mat = []
    for man in maneuvers:
        row = []
        for mod in available_models:
            sel = sub_df[(sub_df["maneuver"] == man) & (sub_df["model"] == mod)]
            if len(sel):
                row.append(float(sel.iloc[0][metric]))
            else:
                row.append(np.nan)
        mat.append(row)
    return np.array(mat)


def plot_metric_grid(metric: str, save_path: Path):
    """
    3 列子图（1s/2s/3s），每格：x=man，分组柱=不同模型，y=metric。
    """
    n_cols = len(horizons)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), sharey=True)

    # 若只有一个地平线，axes 不是数组
    if n_cols == 1:
        axes = [axes]

    bar_width = 0.16
    x = np.arange(len(maneuvers))

    for ax, H in zip(axes, horizons):
        sub = df[df["horizon_s"] == H].copy()
        vals = _values_for_panel(sub, metric)  # (n_manu, n_model)

        for j, mod in enumerate(available_models):
            # 每个模型一个偏移
            offset = (j - (len(available_models) - 1) / 2.0) * bar_width
            ax.bar(x + offset, vals[:, j], width=bar_width, label=mod)

        ax.set_title(f"{metric} @ {int(H)}s")
        ax.set_xticks(x, ["straight", "left", "right"])
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    axes[0].set_ylabel(metric)
    # 统一图例放到底部
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(available_models), frameon=False
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])  # 给底部图例留空间
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


# -----------------------
# Draw
# -----------------------
plot_metric_grid("ADE", OUT_DIR / "summary_ADE.png")
plot_metric_grid("FDE", OUT_DIR / "summary_FDE.png")
