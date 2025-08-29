import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import cv2


path = Path("csv_out")
# 加载CSV文件
df = pd.read_csv(path / "tracking_data.csv")

# 按照车辆ID和时间排序
df.sort_values(by=["id", "frame"], inplace=True)

vehicle_id = 1


def plot_traj_on_img(df, id):
    # 读取视频的第一帧
    video_path = "one_video/DJI_0007.mp4"  # 将 'your_video.mp4' 替换为你的实际文件名
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    df = df[df["id"] == id]

    if ret:
        # 将 BGR 图像转换为 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 在 Matplotlib 中显示图像
        plt.imshow(frame_rgb)

        # 绘制 scatter plot 叠加在图像上
        plt.scatter(df["center_x"], df["center_y"], color="pink", s=0.5)

        plt.axis("on")  # 如果你不想显示坐标轴
        # plt.show()
        plt.savefig("traj_on_img.png")
    else:
        print("cannot read video, pls cheack video dir")


def get_tra_from_id(df, id):
    return df[df["id"] == id]


def plot_turning_per_sec(df, vehicle_id=vehicle_id):
    # 计算每个车辆在每个时间点的位移
    df["delta_x"] = df.groupby("id")["center_x"].diff()
    df["delta_y"] = df.groupby("id")["center_y"].diff()

    # 计算每个点的方向角（弧度），范围为 [-π, π]
    df["theta"] = np.arctan2(df["delta_y"], df["delta_x"])

    # 计算方向角的变化量（角速度）
    df["delta_theta"] = df.groupby("id")["theta"].diff()

    # 为了处理角度跳变的问题，将变化量限制在 [-π, π] 范围内
    df["delta_theta"] = (df["delta_theta"] + np.pi) % (2 * np.pi) - np.pi

    # 填充缺失值
    df["delta_theta"] = df["delta_theta"].fillna(0)

    # 设置角度变化累积的窗口大小，例如3秒（假设帧率为30帧/秒）
    window_size = 6  # 3秒

    # 对角度变化进行累积和
    df["cum_delta_theta"] = (
        df.groupby("id")["delta_theta"]
        .rolling(window_size)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # 设置窗口大小（可以根据需要调整）
    window_size = 10

    # 计算每个点的瞬时方向变化
    df["delta_theta"] = df["theta"].diff().fillna(0)

    # 使用滑动窗口来计算累积方向变化
    df["windowed_delta_theta"] = (
        df["delta_theta"].rolling(window=window_size, min_periods=1).sum()
    )

    # 设置阈值来判断转弯方向
    turn_threshold = np.deg2rad(15)  # 30度的阈值

    # 初始化转弯标签
    df["turn_label"] = "straight"  # 默认为直行

    # 判断左转
    df.loc[df["windowed_delta_theta"] > turn_threshold, "turn_label"] = "left_turn"

    # 判断右转
    df.loc[df["windowed_delta_theta"] < -turn_threshold, "turn_label"] = "right_turn"

    # 定义颜色映射，给 left_turn 和 right_turn 分配鲜艳的颜色
    color_map = {"left_turn": "red", "right_turn": "green", "straight": "pink"}

    # 假设 get_tra_from_id 是获取车辆轨迹数据的函数
    vehicle_df = get_tra_from_id(df, vehicle_id)

    plt.figure(figsize=(10, 8))

    # 绘制轨迹
    plt.plot(vehicle_df["center_x"], vehicle_df["center_y"], "b.-", label="traj")

    # 根据转弯标签改变颜色
    for label, group in vehicle_df.groupby("turn_label"):
        plt.plot(
            group["center_x"],
            group["center_y"],
            ".",
            label=label,
            color=color_map.get(label, "blue"),
        )

    plt.gca().invert_yaxis()  # 反转 y 轴以匹配 OpenCV 的坐标系
    plt.legend()
    plt.xlabel("center x")
    plt.ylabel("center y")
    plt.title(f"vehicle {vehicle_id} trajectory and turning label at each time")

    plot_traj_on_img(df, vehicle_id)
    plt.savefig("turning_label.png")

    # plt.show()
    #


ego_vehicle_df = get_tra_from_id(df, vehicle_id)
ego_vehicle_df.head()
ego_vehicle_df.frame.max() / 30

ego_vehicle_df.describe()
ego_vehicle_df.info()
# calculate rurning label
plot_turning_per_sec(df, vehicle_id)
# plot_traj_on_img(df, vehicle_id)
# plt.show()
