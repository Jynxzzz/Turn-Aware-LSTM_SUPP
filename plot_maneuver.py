import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("results_by_maneuver.csv")
OUT_DIR = Path("figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 读取
df = pd.read_csv(CSV_PATH)

# 只保留我们关心的顺序与标签
models_order = ["CV", "VanillaLSTM", "TurnAwareLSTM", "TinyTransformer"]
maneuvers = ["straight", "left_turn", "right_turn"]
horizons = [1.0, 2.0, 3.0]  # 秒

# 检查可用模型
available_models = [m for m in models_order if m in df["model"].unique()]


def plot_grouped_bars(metric: str, horizon_s: float):
    """
    metric: 'ADE' or 'FDE'
    horizon_s: 1.0 / 2.0 / 3.0
    画一张图：x 轴为 maneuver，柱为不同模型的分组柱状
    """
    sub = df[df["horizon_s"] == horizon_s].copy()
    # 按照 maneuver × model 取值（保持顺序）
    values = []
    for man in maneuvers:
        row_vals = []
        for mod in available_models:
            sel = sub[(sub["maneuver"] == man) & (sub["model"] == mod)]
            if len(sel):
                row_vals.append(float(sel[metric].values[0]))
            else:
                row_vals.append(np.nan)
        values.append(row_vals)

    values = np.array(values)  # shape = (len(maneuvers), len(models))
    x = np.arange(len(maneuvers))
    width = 0.18  # 柱宽

    plt.figure(figsize=(9, 5))
    for i, mod in enumerate(available_models):
        # 每个模型一个偏移
        plt.bar(
            x + (i - (len(available_models) - 1) / 2) * width,
            values[:, i],
            width=width,
            label=mod,
        )

    plt.xticks(x, maneuvers)
    plt.ylabel(metric)
    plt.title(f"{metric} by Maneuver @ {horizon_s:.0f}s")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    fname = OUT_DIR / f"fig_{metric.lower()}_{int(horizon_s)}s.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved: {fname}")


# 依次输出 6 张图（ADE/FDE × 1s/2s/3s）
for H in horizons:
    plot_grouped_bars("ADE", H)
    plot_grouped_bars("FDE", H)
