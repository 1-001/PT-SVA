import pandas as pd
import numpy as np

# 读取Excel文件
file_path = 'train.xlsx'
df = pd.read_excel(file_path)

# 检查Base Severity列的唯一值
categories = df['Base Severity'].unique()

# 创建一个字典来保存分割的数据集
datasets = {f'dataset_{i}': pd.DataFrame() for i in range(1, 5)}

# 定义抽样比例
ratios = [0.2, 0.4, 0.6, 0.8]

# 对每个类别分别处理
for category in categories:
    category_df = df[df['Base Severity'] == category]

    for i, ratio in enumerate(ratios):
        sample_size = int(len(category_df) * ratio)
        sample_df = category_df.sample(sample_size, random_state=42)
        datasets[f'dataset_{i + 1}'] = pd.concat([datasets[f'dataset_{i + 1}'], sample_df])

# 保存每个数据集到Excel文件
for i in range(1, 5):
    datasets[f'dataset_{i}'].to_excel(f'dataset_{i}.xlsx', index=False)
