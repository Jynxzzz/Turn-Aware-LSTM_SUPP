# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python (drone_detection)
#     language: python
#     name: drone_detection
# ---

# %%
import pandas as pd
import numpy as np
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# %%
path = Path('csv_out')
# 加载CSV文件
df = pd.read_csv(path/'tracking_data.csv')

# 按照车辆ID和时间排序
df.sort_values(by=['id', 'frame'], inplace=True)

# %%
df.head()

# %%

# %%
df =  df.groupby('id').apply(lambda x: x.iloc[::30]).reset_index(drop=True)


# %%

# %%

def get_tra_from_id(df, vehicle_id):
    return df[df['id'] == vehicle_id]

def plot_turning_per_sec(df, vehicle_id, deg_thresh=10):
    df['center_y_inverted'] = df['center_y'] * -1
    


    
    # 计算每个车辆在每个时间点的位移
    df.loc[:, 'delta_x'] = df.groupby('id')['center_x'].diff()
    df.loc[:, 'delta_y'] = df.groupby('id')['center_y_inverted'].diff()
    
    # 计算每个点的方向角（弧度），范围为 [-π, π]
    df.loc[:, 'theta'] = np.arctan2(df['delta_y'], df['delta_x'])

    # 计算方向角的变化量（角速度）
    df.loc[:, 'delta_theta'] = df.groupby('id')['theta'].diff()
    
    # 为了处理角度跳变的问题，将变化量限制在 [-π, π] 范围内
    df.loc[:, 'delta_theta'] = (df['delta_theta'] + np.pi) % (2 * np.pi) - np.pi
    
    # 填充缺失值
    df.loc[:, 'delta_theta'] = df['delta_theta'].fillna(0)
    
    # 设置角度变化累积的窗口大小，例如3秒（假设帧率为30帧/秒）
    window_size = 2  # 2秒 
    
    # 对角度变化进行累积和
    df.loc[:, 'cum_delta_theta'] = df.groupby('id')['delta_theta'].rolling(window_size).sum().reset_index(level=0, drop=True)
    
    # 设置窗口大小（可以根据需要调整）
    
    
    # 使用滑动窗口来计算累积方向变化
    df.loc[:, 'windowed_delta_theta'] = df['delta_theta'].rolling(window=window_size, min_periods=1).sum()
    
    # 设置阈值来判断转弯方向
    turn_threshold = np.deg2rad(deg_thresh)  # 15度的阈值
    
    # 初始化转弯标签
    df['turn_label'] = 'Forward'  # 默认为直行
    
    # 计算像素上的移动距离
    df['distance_moved'] = np.sqrt(df['delta_x'] ** 2 + df['delta_y'] ** 2)
    
    # 将像素转换为米，1个像素对应 3.5/19 米
    df['distance_moved_meters'] = df['distance_moved'] * (3.5 / 19)
    
    # 设置移动距离的阈值，低于该阈值表示车辆没有实际移动
    movement_threshold = 0.5  # 阈值设为0.5米（可以根据实际情况调整）
    
    # 判断左转，仅当车辆实际移动的距离超过阈值时才标记为转弯
    df.loc[(df['windowed_delta_theta'] > turn_threshold) & (df['distance_moved_meters'] > movement_threshold), 'turn_label'] = 'Turn Left'
    
    # 判断右转
    df.loc[(df['windowed_delta_theta'] < -turn_threshold) & (df['distance_moved_meters'] > movement_threshold), 'turn_label'] = 'Turn Right'
    
    # 定义颜色映射，给 left_turn 和 right_turn 分配鲜艳的颜色
    color_map = {
        'Turn Left': 'red',
        'Turn Right': 'pink',
        'Forward': 'yellow'
    }
    
    # 获取车辆轨迹数据
    # vehicle_df = get_tra_from_id(df, vehicle_id)
    vehicle_df = df
    
    plt.figure(figsize=(10, 8))
    
    # 绘制轨迹
    plt.plot(vehicle_df['center_x'], vehicle_df['center_y'], 'b.-', label='Trajectory')
    
    # 根据转弯标签改变颜色
    for label, group in vehicle_df.groupby('turn_label'):
        plt.plot(group['center_x'], group['center_y'], '.', label=label, color=color_map.get(label, 'gray'))
    
    plt.gca().invert_yaxis()  # 反转 y 轴以匹配 OpenCV 的坐标系
    plt.legend()
    plt.xlabel('Center x')
    plt.ylabel('Center y')
    # plt.title(f'vehicle {vehicle_id} trajectory and turning label at each time')
    return df
    
    # plt.show() 


# %%

# %%

# %%

# %%
def plot_background_img():
    
    # 读取视频的第一帧
    video_path = 'one_video/DJI_0007.mp4'  # 将 'your_video.mp4' 替换为你的实际文件名
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
        
        
        plt.axis('on')  # 如果你不想显示坐标轴
        plt.savefig('background.png', dpi=300)
                
        # plt.savefig(f'background.jpeg', dpi=300)
        # plt.show()
        
    else:
        print("cannot read video, pls cheack video dir")

plot_background_img()

# %%

# %%
# vehicle_id = 50
vehicle_id = np.random.choice([50, 328, 220, 46, 201, 238, 278, 185, 309, 303, 74, 93, 127, 203, 219, 210, 280, 390])


target_df = get_tra_from_id(df, vehicle_id)

turning_target_df = plot_turning_per_sec(target_df, vehicle_id)
plot_background_img()
plt.savefig(f'vehicle {vehicle_id} trajectory and turning label at each time', dpi=300)
plt.show()



# %%
# 绘制5个不同车辆的轨迹图
for _ in range(5):
    
    vehicle_id = np.random.choice(df.id.unique())
    target_df = get_tra_from_id(df, vehicle_id)
    plot_turning_per_sec(target_df, vehicle_id)
    plot_background_img()


# %%

# %%

# %%
# 50, 328, 220, 46, 201, 238, 278, 185, 309, 303, 74, 93, 127, 203, 219, 210, 280, 390

# %%

# %%

# %%

# %%
def refine_turning_label(df, vehicle_id, threshold=3):
    # 先对特定车辆的所有turn_label进行判断
    vehicle_df = df[df['id'] == vehicle_id].copy()

    # 计算每种转弯标签的数量
    left_turn_count = (vehicle_df['turn_label'] == 'left_turn').sum()
    right_turn_count = (vehicle_df['turn_label'] == 'right_turn').sum()

    # 如果 left_turn 的数量大于阈值，整体标记为 left_turn
    if left_turn_count >= threshold:
        vehicle_df['overall_turn_label'] = 'left_turn'
    # 如果没有足够的 left_turn，但 right_turn 的数量大于阈值，整体标记为 right_turn
    elif right_turn_count >= threshold:
        vehicle_df['overall_turn_label'] = 'right_turn'
    # 否则保持为 straight
    else:
        vehicle_df['overall_turn_label'] = 'straight'

    return vehicle_df


# %%
# vehicle_id = np.random.choice([50, 328, 220, 46, 201, 238, 278, 185, 309, 303, 74, 93, 127, 203, 219, 210, 280, 390])
vehicle_id = 50
target_df = get_tra_from_id(df, vehicle_id)

turning_target_df = plot_turning_per_sec(target_df, vehicle_id)
plot_background_img()
plt.show()

# Example usage
refined_df = refine_turning_label(turning_target_df, vehicle_id)

print(refined_df['overall_turn_label'])

print(turning_target_df.turn_label.value_counts())


# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
# 50, 328, 220, 46, 201, 238, 278, 185, 309, 303, 74, 93, 127, 203, 219, 210, 280, 390

# %%
turning_target_df

# %%
# 绘制5个不同车辆的轨迹图
for _ in range(5):
    
    vehicle_id = np.random.choice([50, 328, 220, 46, 201, 238, 278, 185, 309, 303, 74, 93, 127, 203, 219, 210, 280, 390])
    target_df = get_tra_from_id(df, vehicle_id)
    plot_turning_per_sec(target_df, vehicle_id)
    plot_background_img()

# %%


def calc_turning_per_sec(df, vehicle_id, deg_thresh=10):
    df['center_y_inverted'] = df['center_y'] * -1
    


    
    # 计算每个车辆在每个时间点的位移
    df.loc[:, 'delta_x'] = df.groupby('id')['center_x'].diff()
    df.loc[:, 'delta_y'] = df.groupby('id')['center_y_inverted'].diff()
    
    # 计算每个点的方向角（弧度），范围为 [-π, π]
    df.loc[:, 'theta'] = np.arctan2(df['delta_y'], df['delta_x'])

    # 计算方向角的变化量（角速度）
    df.loc[:, 'delta_theta'] = df.groupby('id')['theta'].diff()
    
    # 为了处理角度跳变的问题，将变化量限制在 [-π, π] 范围内
    df.loc[:, 'delta_theta'] = (df['delta_theta'] + np.pi) % (2 * np.pi) - np.pi
    
    # 填充缺失值
    df.loc[:, 'delta_theta'] = df['delta_theta'].fillna(0)
    
    # 设置角度变化累积的窗口大小，例如3秒（假设帧率为30帧/秒）
    window_size = 2  # 2秒 
    
    # 对角度变化进行累积和
    df.loc[:, 'cum_delta_theta'] = df.groupby('id')['delta_theta'].rolling(window_size).sum().reset_index(level=0, drop=True)
    
    # 设置窗口大小（可以根据需要调整）
    
    
    # 使用滑动窗口来计算累积方向变化
    df.loc[:, 'windowed_delta_theta'] = df['delta_theta'].rolling(window=window_size, min_periods=1).sum()
    
    # 设置阈值来判断转弯方向
    turn_threshold = np.deg2rad(deg_thresh)  # 15度的阈值
    
    # 初始化转弯标签
    df['turn_label'] = 'straight'  # 默认为直行
    
    # 计算像素上的移动距离
    df['distance_moved'] = np.sqrt(df['delta_x'] ** 2 + df['delta_y'] ** 2)
    
    # 将像素转换为米，1个像素对应 3.5/19 米
    df['distance_moved_meters'] = df['distance_moved'] * (3.5 / 19)
    
    # 设置移动距离的阈值，低于该阈值表示车辆没有实际移动
    movement_threshold = 0.5  # 阈值设为0.5米（可以根据实际情况调整）
    
    # 判断左转，仅当车辆实际移动的距离超过阈值时才标记为转弯
    df.loc[(df['windowed_delta_theta'] > turn_threshold) & (df['distance_moved_meters'] > movement_threshold), 'turn_label'] = 'left_turn'
    
    # 判断右转
    df.loc[(df['windowed_delta_theta'] < -turn_threshold) & (df['distance_moved_meters'] > movement_threshold), 'turn_label'] = 'right_turn'

    return df
    
    # plt.show() 


# %%
def process_all_vehicle_ids(df):
    all_refined_dfs = []  # 用于存储所有处理后的数据
    
    # 按id分组
    for vehicle_id, group in df.groupby('id'):
        print(f"Processing vehicle_id: {vehicle_id}")
        
        # 获取当前车辆的数据
        target_df = get_tra_from_id(df, vehicle_id)
        
        # 计算转弯标签
        turning_target_df = calc_turning_per_sec(target_df, vehicle_id)
        
        # 应用 refine_turning_label 来生成精简后的转弯标签
        refined_df = refine_turning_label(turning_target_df, vehicle_id)
        
        # 将处理后的结果添加到列表中
        all_refined_dfs.append(refined_df)
    
    # 将所有处理后的数据框合并
    final_df = pd.concat(all_refined_dfs, ignore_index=True)
    
    return final_df


# Example usage:
final_df = process_all_vehicle_ids(df)

# 打印所有处理后的转弯标签统计
print(final_df['turn_label'].value_counts())

# 可以选择进一步的操作，比如保存或可视化

# %%

# %%
final_df.head()

# %%
final_df.to_csv('csv_out/overall_turn_label.csv', index=False)

# %%

# %%

# %%
