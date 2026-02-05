import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# ==========================================
# 1. 定义可配置变量
# ==========================================
# 颜色定义 (RGB 0-1)
# 排名第一的颜色 (Mint Green)
rank1_color = (0.6, 1.0, 0.6) 
# 排名最后的颜色 (Light Purple)
rankl_color = (0.8, 0.6, 1.0) 

# Season 区间
season_start = 1
season_final = 34 # 假设最大到34，代码中会根据数据调整

# ==========================================
# 2. 数据读取与预处理
# ==========================================
# 假设脚本在 cur代码 目录下，数据在上一级目录
data_path = '../C_origin.csv'
if not os.path.exists(data_path):
    # 尝试当前目录 (以防运行位置不同)
    data_path = 'C_origin.csv'

try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: Data file not found at {data_path}")
    exit()

# 筛选 Season
# 确保 season 列是数值型
df['season'] = pd.to_numeric(df['season'], errors='coerce')
df = df.dropna(subset=['season'])
df = df[(df['season'] >= season_start) & (df['season'] <= season_final)]

# 提取需要的列
target_cols = ['celebrity_industry', 'celebrity_homecountry/region', 'celebrity_age_during_season', 'placement']
# 检查列是否存在
missing_cols = [c for c in target_cols if c not in df.columns]
if missing_cols:
    # 尝试修正列名 (处理 potential naming mismatches)
    rename_map = {
        'celebrity_homecountry_region': 'celebrity_homecountry/region',
        'celebrity_homecountry': 'celebrity_homecountry/region',
        'age_during_season': 'celebrity_age_during_season'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # 再次检查
    missing_cols = [c for c in target_cols if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        print("Available columns:", df.columns.tolist())
        exit()

# 数据清洗
df = df[target_cols].dropna()

# 转换数据类型
df['celebrity_age_during_season'] = pd.to_numeric(df['celebrity_age_during_season'], errors='coerce')
df['placement'] = pd.to_numeric(df['placement'], errors='coerce')
df = df.dropna() # 再次去除转换失败的行

# 将分类变量转换为数值编码 (X, Y轴)
# Industry
unique_industries = sorted(df['celebrity_industry'].unique())
industry_map = {val: i for i, val in enumerate(unique_industries)}
df['industry_code'] = df['celebrity_industry'].map(industry_map)

# Homecountry
unique_countries = sorted(df['celebrity_homecountry/region'].unique())
country_map = {val: i for i, val in enumerate(unique_countries)}
df['country_code'] = df['celebrity_homecountry/region'].map(country_map)

# ==========================================
# 3. 颜色映射逻辑
# ==========================================
# 归一化 placement
# 1 -> 0.0 (接近 rank1_color)
# Max -> 1.0 (接近 rankl_color)
max_rank = df['placement'].max()
min_rank = df['placement'].min() # 应该是1

def get_color(rank):
    if max_rank == min_rank:
        return rank1_color
    
    # t: 0.0 (Best/1st) -> 1.0 (Worst/Last)
    t = (rank - min_rank) / (max_rank - min_rank)
    
    # Linear interpolation
    r = rank1_color[0] + (rankl_color[0] - rank1_color[0]) * t
    g = rank1_color[1] + (rankl_color[1] - rank1_color[1]) * t
    b = rank1_color[2] + (rankl_color[2] - rank1_color[2]) * t
    
    return (r, g, b)

colors = df['placement'].apply(get_color).tolist()

# ==========================================
# 4. 3D 可视化
# ==========================================
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
# X: Industry, Y: Country, Z: Age
ax.scatter(df['industry_code'], df['country_code'], df['celebrity_age_during_season'], c=colors, s=50, alpha=0.8)

# 设置轴标签
ax.set_xlabel('Industry')
ax.set_ylabel('Home Country/Region')
ax.set_zlabel('Age')

# 设置刻度标签 (如果太多可能会重叠，这里做个简单处理)
# X轴刻度
if len(unique_industries) > 10:
    # 太多了，只显示部分或者旋转
    ax.set_xticks(range(len(unique_industries)))
    ax.set_xticklabels(unique_industries, rotation=45, ha='right', fontsize=8)
else:
    ax.set_xticks(range(len(unique_industries)))
    ax.set_xticklabels(unique_industries, rotation=45, ha='right')

# Y轴刻度
if len(unique_countries) > 10:
    ax.set_yticks(range(len(unique_countries)))
    ax.set_yticklabels(unique_countries, rotation=-15, ha='left', fontsize=8)
else:
    ax.set_yticks(range(len(unique_countries)))
    ax.set_yticklabels(unique_countries, rotation=-15)

plt.title(f'3D Characteristic Analysis (Season {season_start}-{season_final})\nColor: Green(1st) -> Purple(Last)')
plt.tight_layout()

# 保存
output_file = '3_dim_characterastic.png'
plt.savefig(output_file, dpi=300)
print(f"Successfully saved plot to {output_file}")
