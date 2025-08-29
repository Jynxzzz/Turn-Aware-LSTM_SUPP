import pandas as pd

df1 = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 31, 32, 33, 34, 34, 36]})

df1

df2 = pd.DataFrame({"id": [ 3, 5, 7, 9, 11], "label": [1, 2, 3, 4, 5]})

df2

# merged_df = pd.merge_asof(df1, df2, on="id", direction="forward")
merged_df = pd.merge_asof(df1, df2, on="id", direction="backward")
# merged_df = pd.merge_asof(df1, df2, on="id", direction="nearest")
merged_df

