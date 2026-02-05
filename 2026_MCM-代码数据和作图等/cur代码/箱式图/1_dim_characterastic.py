import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.interpolate import make_interp_spline

# ==========================================
# 1. 定义可配置变量
# ==========================================
season_start = 1
season_final = 34

# ==========================================
# 2. 数据读取与预处理
# ==========================================
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
# 3. 数据归类处理
# ==========================================

# 3.1 国家/地区 -> 大洲 映射
# 基于 C_origin.csv 中出现的国家进行手动映射
# 如果有未列出的国家，默认归为 'Others'
country_to_continent = {
    'United States': 'North America',
    'New Zealand': 'Oceania',
    'England': 'Europe',
    'Mexico': 'North America',
    'Canada': 'North America',
    'Czechoslovakia': 'Europe',
    'Brazil': 'South America',
    'Chile': 'South America',
    'Yugoslavia': 'Europe',
    'France': 'Europe',
    'Australia': 'Oceania',
    'Russia': 'Europe', # Or Asia, usually Europe culturally in DWTS context
    'Ukraine': 'Europe',
    'Poland': 'Europe',
    'Italy': 'Europe',
    'Ireland': 'Europe',
    'Cuba': 'North America',
    'Israel': 'Asia',
    'Philippines': 'Asia',
    'India': 'Asia',
    'South Africa': 'Africa',
    'Germany': 'Europe',
    'Sweden': 'Europe',
    'Latvia': 'Europe',
    'Albania': 'Europe',
    'Slovenia': 'Europe',
    'Colombia': 'South America',
    'Argentina': 'South America',
    'Panama': 'North America',
    'Puerto Rico': 'North America',
    'Croatia': 'Europe',
    'South Korea': 'Asia',
    'Spain': 'Europe',
    'Taiwan China': 'Asia',
    'Venezuela': 'South America',
    'Wales': 'Europe'
}

def map_country(country):
    if not isinstance(country, str):
        return 'Others'
    # 处理可能的空格
    country = country.strip()
    return country_to_continent.get(country, 'Others')

df['region_group'] = df['celebrity_homecountry/region'].apply(map_country)

# 3.2 年龄 -> 年龄段 (每10岁)
# 20-29, 30-39, etc.
def map_age_group(age):
    try:
        age = int(age)
        lower = (age // 10) * 10
        upper = lower + 9
        return f"{lower}-{upper}"
    except:
        return "Unknown"

df['age_group'] = df['celebrity_age_during_season'].apply(map_age_group)

# 排序年龄组，保证图表顺序
unique_age_groups = sorted(df['age_group'].unique())

# ==========================================
# 4. 绘图函数
# ==========================================
def create_boxplot(x_col, y_col, output_filename, x_label, y_label, order=None, add_trend=False, invert_y=False, palette="Set3"):
    plt.figure(figsize=(14, 8))
    
    # 使用 seaborn 绘制箱式图
    # whis=1.5 是默认的 IQR 倍数，boxprops 并不直接控制显示的百分比，
    # 但箱体(Box)本身定义就是 IQR (25%-75%)。
    # 用户要求：中间箱体部分仅展示数据的75%区间（四分位范围） -> 这就是标准箱式图的定义 (Q1 to Q3)
    
    sns.boxplot(x=x_col, y=y_col, data=df, order=order, palette=palette, showfliers=True)
    
    # 添加平滑均值连线 (针对所有图表)
    if add_trend and order is not None:
        # 计算每个组的平均值
        means = df.groupby(x_col)[y_col].mean().reindex(order)
        
        # 准备数据，移除 NaN
        x_coords = np.arange(len(order))
        valid_mask = ~means.isna()
        
        if valid_mask.sum() > 1:
            x_valid = x_coords[valid_mask]
            y_valid = means[valid_mask]
            
            # 尝试使用 spline 插值生成平滑曲线
            # 如果点数太少 (例如 2 个)，spline k=3 会失败，降级处理
            try:
                # 只有当点数 >= 4 时才使用 k=3 (三次样条)
                # 点数 2 或 3 时分别使用 k=1 或 k=2
                n_points = len(x_valid)
                k_order = 3 if n_points >= 4 else (n_points - 1)
                
                # 生成平滑的 X 坐标
                x_smooth = np.linspace(x_valid.min(), x_valid.max(), 300)
                
                # 生成插值模型
                spl = make_interp_spline(x_valid, y_valid, k=k_order)
                y_smooth = spl(x_smooth)
                
                # 绘制黑色平滑曲线
                plt.plot(x_smooth, y_smooth, color='black', linestyle='-', linewidth=2.5, label='Mean Trend')
                
            except Exception as e:
                print(f"Smoothing failed, falling back to linear: {e}")
                # 回退到普通折线
                plt.plot(x_valid, y_valid, color='black', marker='o', linestyle='-', linewidth=2.5, label='Mean Trend')
        
        # 不再绘制多项式拟合曲线 (Poly Fit)
        # 仅保留上述平滑连线

    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(f'{x_label} vs {y_label} (Season {season_start}-{season_final})', fontsize=14)
    
    # 旋转X轴标签以防重叠
    plt.xticks(rotation=45, ha='right')
    
    # 反转 Y 轴 (针对 Age 图)
    if invert_y:
        plt.gca().invert_yaxis()
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # 保存
    save_path = output_filename
    # 如果脚本在 cur代码 目录下运行，直接保存；否则拼接路径
    if os.path.basename(os.getcwd()) != 'cur代码':
         if not os.path.exists('cur代码'):
             os.makedirs('cur代码')
         save_path = os.path.join('cur代码', output_filename)
         
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")

# 自定义颜色调色板 (用户指定)
custom_palette = [
    (77/255, 103/255, 164/255),   # rank1
    (140/255, 141/255, 197/255),  # rank2
    (186/255, 168/255, 210/255),  # rank3
    (237/255, 187/255, 199/255),  # rank4
    (255/255, 204/255, 154/255),  # rank5
    (246/255, 162/255, 126/255),  # rank6
    (189/255, 115/255, 106/255),  # rank7
    (121/255, 77/255, 72/255)     # rank8
]

# ==========================================
# 5. 生成三个图表
# ==========================================

# 1. Industry vs Placement
# Industry 归类
# 保留指定的 6 类，其余归为 'Others'
target_industries = [
    'Actor/Actress', 
    'Athlete', 
    'Comedian', 
    'Model', 
    'Singer/Rapper', 
    'TV Personality'
]

def map_industry(ind):
    if not isinstance(ind, str):
        return 'Others'
    ind = ind.strip()
    if ind in target_industries:
        return ind
    return 'Others'

df['industry_group'] = df['celebrity_industry'].apply(map_industry)

# 按中位数排序以便观察
industry_order = df.groupby('industry_group')['placement'].median().sort_values().index
create_boxplot('industry_group', 'placement', 'industry.png', 'Celebrity Industry', 'Placement (Higher is Top)', 
               order=industry_order, invert_y=True, palette=custom_palette, add_trend=True)

# 2. Region vs Placement
# 按 Region 归类
region_order = df.groupby('region_group')['placement'].median().sort_values().index
create_boxplot('region_group', 'placement', 'homeregion.png', 'Home Region (Continent)', 'Placement (Higher is Top)', 
               order=region_order, invert_y=True, palette=custom_palette, add_trend=True)

# 3. Age vs Placement
# 按年龄段归类，保持自然顺序 (10-19, 20-29...)
age_order = sorted(df['age_group'].unique())

# 针对 Age 图：增加拟合曲线，反转 Y 轴，使用自定义颜色
create_boxplot('age_group', 'placement', 'age.png', 'Age Group', 'Placement (Higher is Top)', 
               order=age_order, add_trend=True, invert_y=True, palette=custom_palette)
