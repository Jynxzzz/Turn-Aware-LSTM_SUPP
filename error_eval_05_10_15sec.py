#!/usr/bin/env python
# coding: utf-8

# In[1]:

import random
import joblib
from torch.utils.data import DataLoader, TensorDataset
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

# In[2]:

path = Path("csv_out")
eval_video_path = Path("eval_model_on_video")

# In[3]:

# 读取数据
df1 = pd.read_csv(path / "tracking_data.csv")
df2 = pd.read_csv(path / "overall_turn_label.csv")

# 合并数据帧
df_merged = pd.merge(
    df1, df2[["id", "frame", "overall_turn_label"]], on=["id", "frame"], how="left"
)

# 按 id 分组，对 overall_turn_label 进行前向和后向填充
df_merged["overall_turn_label"] = df_merged.groupby("id")["overall_turn_label"].fillna(
    method="ffill"
)
df_merged["overall_turn_label"] = df_merged.groupby("id")["overall_turn_label"].fillna(
    method="bfill"
)

# 检查是否仍有缺失值
missing_values = df_merged["overall_turn_label"].isnull().sum()
print(f"缺失的 overall_turn_label 数量：{missing_values}")

# 如果仍有缺失值，可以选择填充默认值或删除这些行
df_merged["overall_turn_label"] = df_merged["overall_turn_label"].fillna("straight")

# 对 overall_turn_label 进行 One-Hot 编码
encoder = OneHotEncoder(sparse_output=False)
turn_labels_encoded = encoder.fit_transform(df_merged[["overall_turn_label"]])
turn_label_columns = encoder.get_feature_names_out(["overall_turn_label"])
df_merged[turn_label_columns] = turn_labels_encoded

# 定义输入特征
input_features = ["center_x", "center_y"] + list(turn_label_columns)

# 定义序列长度
sequence_length = 90  # 输入序列长度（90帧，相当于3秒的历史数据）
predict_length = 45  # 输出序列长度（45帧，相当于1.5秒的预测）

# 生成输入和目标序列
input_sequences = []
target_sequences = []
sequence_vehicle_ids = []

grouped = df_merged.groupby("id")

for track_id, group in grouped:
    group = group.sort_values("frame").reset_index(drop=True)
    features = group[input_features].values

    num_sequences = len(features) - sequence_length - predict_length + 1
    if num_sequences <= 0:
        continue

    for i in range(num_sequences):
        input_seq = features[i : i + sequence_length]
        # 只取 center_x 和 center_y
        target_seq = features[
            i + sequence_length : i + sequence_length + predict_length, :2
        ]

        input_sequences.append(input_seq)
        target_sequences.append(target_seq)
        sequence_vehicle_ids.append(track_id)

# 转换为 NumPy 数组
input_sequences = np.array(input_sequences)
target_sequences = np.array(target_sequences)
sequence_vehicle_ids = np.array(sequence_vehicle_ids)

# 数据标准化
numeric_feature_indices = [0, 1]  # 'center_x', 'center_y'

all_numeric_inputs = input_sequences[:, :, numeric_feature_indices].reshape(
    -1, len(numeric_feature_indices)
)

scaler = StandardScaler()
scaler.fit(all_numeric_inputs)

input_sequences[:, :, numeric_feature_indices] = scaler.transform(
    all_numeric_inputs
).reshape(
    input_sequences.shape[0], input_sequences.shape[1], len(numeric_feature_indices)
)

all_numeric_targets = target_sequences.reshape(-1, len(numeric_feature_indices))
target_sequences = scaler.transform(all_numeric_targets).reshape(
    target_sequences.shape[0], target_sequences.shape[1], len(numeric_feature_indices)
)

# In[4]:


# 模型定义


class TrajectoryPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=2):
        super(TrajectoryPredictor, self).__init__()
        self.lstm_encoder = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.lstm_decoder = nn.LSTM(
            output_size, hidden_size, num_layers, batch_first=True
        )
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, target_len):
        batch_size = x.size(0)

        # Encoder：输入为完整的输入特征，包括位置和 One-Hot 编码的转弯标签
        _, (hidden, cell) = self.lstm_encoder(x)

        # Decoder inputs: 使用输入序列的最后一个位置坐标作为初始输入
        decoder_input = x[:, -1, :2].unsqueeze(1)  # 只取 'center_x' 和 'center_y'
        outputs = []

        for t in range(target_len):
            # Decoder step
            out, (hidden, cell) = self.lstm_decoder(decoder_input, (hidden, cell))
            out = self.fc_out(out)
            outputs.append(out.squeeze(1))
            decoder_input = out  # 下一时间步的输入为当前输出的位置坐标

        outputs = torch.stack(outputs, dim=1)
        return outputs


# In[5]:


# 初始化模型
input_size = input_sequences.shape[2]  # 包括所有输入特征
output_size = 2  # 只预测 'center_x' 和 'center_y'
model = TrajectoryPredictor(
    input_size=input_size, hidden_size=128, num_layers=2, output_size=output_size
)

# 转换为张量并移动到设备上
inputs = torch.tensor(input_sequences, dtype=torch.float32)
targets = torch.tensor(target_sequences, dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

inputs = inputs.to(device)
targets = targets.to(device)
model = model.to(device)

# 创建数据集和数据加载器
dataset = TensorDataset(inputs, targets)
batch_size = 1024
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"Total number of samples: {len(dataset)}")
print(f"Number of batches per epoch: {len(data_loader)}")

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
target_len = predict_length  # 预测序列的长度

for epoch in range(num_epochs):
    print(f"Starting epoch {epoch+1}/{num_epochs}")
    model.train()
    total_loss = 0
    for batch_idx, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()

        # 前向传播
        outputs = model(batch_inputs, target_len)

        # 计算损失
        loss = criterion(outputs, batch_targets)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 500 == 0 or (batch_idx + 1) == len(data_loader):
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}")

# In[6]:

# 定义计算指标的函数


def compute_metrics(predictions, targets, horizons):
    metrics = {}
    for horizon in horizons:
        # shape: [num_samples, horizon, 2]
        outputs_at_horizon = predictions[:, :horizon, :]
        targets_at_horizon = targets[:, :horizon, :]

        # Compute errors
        # shape: [num_samples, horizon, 2]
        errors = outputs_at_horizon - targets_at_horizon
        squared_errors = errors**2
        mse = squared_errors.mean().item()
        rmse = np.sqrt(mse)

        abs_errors = errors.abs()
        mae = abs_errors.mean().item()

        # Compute ADE
        # Euclidean distance over x and y
        displacement_errors = torch.norm(errors, dim=2)
        ade = displacement_errors.mean().item()

        # Compute FDE
        final_errors = errors[:, -1, :]  # shape: [num_samples, 2]
        fde = torch.norm(final_errors, dim=1).mean().item()

        metrics[horizon] = {"RMSE": rmse, "MAE": mae, "ADE": ade, "FDE": fde}

    return metrics


# In[7]:


# 在训练集上评估模型
model.eval()
with torch.no_grad():
    total_outputs = []
    total_targets = []
    for batch_inputs, batch_targets in data_loader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        outputs = model(batch_inputs, target_len)
        total_outputs.append(outputs.cpu())
        total_targets.append(batch_targets.cpu())

    total_outputs = torch.cat(total_outputs, dim=0)
    total_targets = torch.cat(total_targets, dim=0)

    # 定义预测的时间地平线
    horizons = {
        15: 0.5,  # 0.5 seconds (15 frames)
        30: 1.0,  # 1.0 seconds (30 frames)
        45: 1.5,  # 1.5 seconds (45 frames)
    }

    # 过滤超过预测长度的地平线
    horizons = {k: v for k, v in horizons.items() if k <= predict_length}

    metrics = compute_metrics(total_outputs, total_targets, horizons.keys())

    for horizon_frames, time_sec in horizons.items():
        print(f"\nMetrics for horizon: {time_sec} seconds ({horizon_frames} frames)")
        print(f"RMSE: {metrics[horizon_frames]['RMSE']:.4f}")
        print(f"MAE: {metrics[horizon_frames]['MAE']:.4f}")
        print(f"ADE: {metrics[horizon_frames]['ADE']:.4f}")
        print(f"FDE: {metrics[horizon_frames]['FDE']:.4f}")

# In[8]:

# 保存模型和 scaler
torch.save(model.state_dict(), eval_video_path / "trajectory_predictor.pth")
joblib.dump(scaler, eval_video_path / "scaler.save")

# In[9]:

# 车辆 ID 映射到索引
vehicle_ids_of_interest = [
    50,
    328,
    220,
    46,
    201,
    238,
    278,
    185,
    309,
    303,
    74,
    93,
    127,
    203,
    219,
    210,
    280,
    390,
]
vehicle_id_to_indices = {}

for vehicle_id in vehicle_ids_of_interest:
    indices = np.where(sequence_vehicle_ids == vehicle_id)[0]
    if len(indices) > 0:
        vehicle_id_to_indices[vehicle_id] = indices
    else:
        print(f"Vehicle ID {vehicle_id} not found in the sequences.")

# In[10]:


def plot_background_img():
    # 读取视频的第一帧
    video_path = "one_video/DJI_0007.mp4"  # 将 'your_video.mp4' 替换为你的实际文件名
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if ret:
        # 将 BGR 图像转换为 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 在 Matplotlib 中显示图像
        plt.imshow(frame_rgb)

        # 绘制 scatter plot 叠加在图像上
        # plt.scatter(df['center_x'], df['center_y'], color='pink', s=0.5)

        plt.axis("on")  # 如果你不想显示坐标轴
        plt.savefig("background.jpeg")
        plt.show()

    else:
        print("Cannot read video, please check video directory")


# In[11]:


# 可视化预测结果
model.eval()
with torch.no_grad():
    # Collect all indices from all vehicle IDs
    all_indices = []
    index_to_vehicle_id = {}
    for vehicle_id, indices in vehicle_id_to_indices.items():
        for idx in indices:
            all_indices.append(idx)
            index_to_vehicle_id[idx] = vehicle_id  # Map index to vehicle ID

    # Randomly select 5 indices
    num_samples = 5
    if len(all_indices) >= num_samples:
        selected_indices = random.sample(all_indices, num_samples)
    else:
        selected_indices = all_indices  # If less than 5 sequences are available

    for idx in selected_indices:
        test_input = inputs[idx].unsqueeze(0).to(device)
        true_target = targets[idx].to(device)

        # Perform prediction
        predicted_output = model(test_input, target_len)

        # Convert predictions and true targets to NumPy arrays
        predicted_output = predicted_output.squeeze(0).cpu().numpy()
        true_target = true_target.cpu().numpy()

        # Get historical input data for visualization
        history_input = test_input.squeeze(0).cpu().numpy()

        # **Inverse scaling**
        # Indices of 'center_x' and 'center_y'
        numeric_feature_indices = [0, 1]

        # Inverse transform historical inputs
        history_input_numeric = history_input[:, numeric_feature_indices]
        history_input_unscaled = scaler.inverse_transform(history_input_numeric)

        # Inverse transform predicted outputs
        predicted_output_unscaled = scaler.inverse_transform(predicted_output)

        # Inverse transform true targets
        true_target_unscaled = scaler.inverse_transform(true_target)

        # **Visualization**
        plt.figure(figsize=(8, 6))

        # Plot historical trajectory
        plt.plot(
            history_input_unscaled[:, 0],
            history_input_unscaled[:, 1],
            "bo-",
            label="Historical Trajectory",
        )

        # Plot true future trajectory
        plt.plot(
            true_target_unscaled[:, 0],
            true_target_unscaled[:, 1],
            "go-",
            label="True Future Trajectory",
        )

        # Plot predicted future trajectory
        plt.plot(
            predicted_output_unscaled[:, 0],
            predicted_output_unscaled[:, 1],
            "ro--",
            label="Predicted Future Trajectory",
        )

        plt.legend()
        plt.xlabel("center_x")
        plt.ylabel("center_y")
        plt.title(
            f"Vehicle {index_to_vehicle_id[idx]} Trajectory Prediction (Sequence Index {idx})"
        )

        # 如果有背景图像绘制函数，可以取消注释以下行
        # plot_background_img()

        plt.show()
