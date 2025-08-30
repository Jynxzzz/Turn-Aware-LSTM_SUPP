import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("results_overall.csv")
OUT_DIR = Path("figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 读取与基本校验
df = pd.read_csv(CSV_PATH)
assert set(["model", "horizon_s", "ADE", "FDE"]).issubset(df.columns)

# 固定顺序，方便和论文一致
models_order = ["CV", "VanillaLSTM", "TurnAwareLSTM", "TinyTransformer"]
df["model"] = pd.Categorical(df["model"], categories=models_order, ordered=True)
df = df.sort_values(["model", "horizon_s"])

# 横轴：秒
horizons = sorted(df["horizon_s"].unique())


def plot_metric(metric: str, ylabel: str, fname: str):
    plt.figure(figsize=(7.5, 5))
    for model in models_order:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        # 保障按 horizon_s 排序取值
        ys = [float(sub[sub["horizon_s"] == h][metric].values[0]) for h in horizons]
        plt.plot(horizons, ys, marker="o", linewidth=2, label=model)

    plt.xticks(
        horizons, [f"{int(h)}s" if h.is_integer() else f"{h:.1f}s" for h in horizons]
    )
    plt.xlabel("Prediction horizon")
    plt.ylabel(ylabel)
    plt.title(f"{metric} vs. horizon (overall)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out = OUT_DIR / fname
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


# 生成两张图
plot_metric("ADE", "ADE (m)", "overall_ADE_vs_horizon.png")
plot_metric("FDE", "FDE (m)", "overall_FDE_vs_horizon.png")
