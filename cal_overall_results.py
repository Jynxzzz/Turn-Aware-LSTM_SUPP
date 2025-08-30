from pathlib import Path

import pandas as pd

# 输入输出路径
CSV_IN = Path("results_by_maneuver.csv")
CSV_OUT = Path("results_overall.csv")

# 读数据
df = pd.read_csv(CSV_IN)

# 聚合：按 model + horizon_s 求 ADE/FDE 的均值
overall = df.groupby(["model", "horizon_s"], as_index=False)[["ADE", "FDE"]].mean()

# 保存
overall.to_csv(CSV_OUT, index=False)
print("Saved:", CSV_OUT)
print(overall)
