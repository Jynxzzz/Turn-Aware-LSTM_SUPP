from pathlib import Path
import pandas as pd


import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

import cv2


import warnings

warnings.filterwarnings("ignore")

path = Path("csv_out")

eval_video_path = Path("eval_model_on_video")


df1 = pd.read_csv(path / "tracking_data.csv")
df1.head()
df2 = pd.read_csv(path / "overall_turn_label.csv")
df2.head()

merged_df = pd.merge(df1, df2, on="id")
