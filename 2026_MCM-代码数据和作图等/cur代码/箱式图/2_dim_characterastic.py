import pandas as pd
import matplotlib.pyplot as plt
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
season_final = 34 

# ==========================================
# 2. 数据读取与预处理
# ==========================================
# 假设脚本在 cur代码 目录下，数据在上一级目录
data_path = '../C_origin.csv'
if not os.path.exists(data_path):
    data_path = 'C_origin.csv'

try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: Data file not found at {data_path}")
    exit()

# 筛选 Season
df['season'] = pd.to_numeric(df['season'], errors='coerce')
df = df.dropna(subset=['season'])
df = df[(df['season'] >= season_start) & (df['season'] <= season_final)]

# 提取需要的列
target_cols = ['celebrity_industry', 'celebrity_homecountry/region', 'celebrity_age_during_season', 'placement']
rename_map = {
    'celebrity_homecountry_region': 'celebrity_homecountry/region',
    'celebrity_homecountry': 'celebrity_homecountry/region',
    'age_during_season': 'celebrity_age_during_season'
}
df.rename(columns=rename_map, inplace=True)

# 检查列
missing_cols = [c for c in target_cols if c not in df.columns]
if missing_cols:
    print(f"Error: Missing columns: {missing_cols}")
    exit()

# 数据清洗
df = df[target_cols].dropna()
df['celebrity_age_during_season'] = pd.to_numeric(df['celebrity_age_during_season'], errors='coerce')
df['placement'] = pd.to_numeric(df['placement'], errors='coerce')
df = df.dropna()

# ==========================================
# 3. 颜色映射逻辑
# ==========================================
max_rank = df['placement'].max()
min_rank = df['placement'].min()

def get_color(rank):
    if max_rank == min_rank:
        return rank1_color
    t = (rank - min_rank) / (max_rank - min_rank)
    r = rank1_color[0] + (rankl_color[0] - rank1_color[0]) * t
    g = rank1_color[1] + (rankl_color[1] - rank1_color[1]) * t
    b = rank1_color[2] + (rankl_color[2] - rank1_color[2]) * t
    return (r, g, b)

colors = df['placement'].apply(get_color).tolist()

# ==========================================
# 4. 绘图函数
# ==========================================
# 为了方便绘图，如果是类别型数据，我们需要生成映射
# Industry
unique_industries = sorted(df['celebrity_industry'].unique())
industry_map = {val: i for i, val in enumerate(unique_industries)}
df['industry_code'] = df['celebrity_industry'].map(industry_map)

# Country
unique_countries = sorted(df['celebrity_homecountry/region'].unique())
country_map = {val: i for i, val in enumerate(unique_countries)}
df['country_code'] = df['celebrity_homecountry/region'].map(country_map)

# 定义要绘制的组合 (x_col, y_col, x_label, y_label, x_ticks, y_ticks, output_name)
plots_config = [
    (
        'industry_code', 'country_code', 
        'Industry', 'Home Country/Region', 
        unique_industries, unique_countries, 
        'industry_country.png'
    ),
    (
        'industry_code', 'celebrity_age_during_season', 
        'Industry', 'Age', 
        unique_industries, None, 
        'industry_age.png'
    ),
    (
        'country_code', 'celebrity_age_during_season', 
        'Home Country/Region', 'Age', 
        unique_countries, None, 
        'country_age.png'
    )
]

output_dir = 'cur代码'
if not os.path.exists(output_dir) and os.path.basename(os.getcwd()) != 'cur代码':
    # 如果当前不在 cur代码 且 文件夹不存在（虽然之前应该创建了）
    # 这里假设如果是直接运行 python cur代码/xxx.py，输出就在当前目录(cur代码)或者相对路径
    # 为了保险，我们直接用文件名，假设脚本运行时cwd是项目根目录，那么需要拼上 cur代码/
    # 或者如果cwd是cur代码，则直接文件名
    pass

# 判断保存路径
def get_save_path(filename):
    if os.path.basename(os.getcwd()) == 'cur代码':
        return filename
    else:
        return os.path.join('cur代码', filename)

for x_col, y_col, x_lbl, y_lbl, x_tick_labels, y_tick_labels, filename in plots_config:
    plt.figure(figsize=(12, 8))
    
    # 增加 jitter (抖动) 以防止点重叠
    x_data = df[x_col]
    y_data = df[y_col]
    
    # 如果是离散的类别代码，加一点随机抖动
    if x_col in ['industry_code', 'country_code']:
        x_data = x_data + np.random.uniform(-0.2, 0.2, size=len(df))
    if y_col in ['industry_code', 'country_code']:
        y_data = y_data + np.random.uniform(-0.2, 0.2, size=len(df))
        
    plt.scatter(x_data, y_data, c=colors, s=50, alpha=0.7)
    
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.title(f'{x_lbl} vs {y_lbl} (Season {season_start}-{season_final})')
    
    # 设置刻度标签
    if x_tick_labels:
        plt.xticks(range(len(x_tick_labels)), x_tick_labels, rotation=45, ha='right', fontsize=8)
    
    if y_tick_labels:
        plt.yticks(range(len(y_tick_labels)), y_tick_labels, fontsize=8)
        
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    save_path = get_save_path(filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")

