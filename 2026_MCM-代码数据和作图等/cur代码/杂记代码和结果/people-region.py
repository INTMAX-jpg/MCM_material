import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. 配置变量与颜色
# ==========================================
season_start = 1
season_final = 34

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

# 映射字典 (复用 1_dim_characterastic.py 的逻辑)
country_to_continent = {
    'United States': 'United States',
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
    'Russia': 'Europe', 
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
    country = country.strip()
    return country_to_continent.get(country, 'Others')

# 应用映射
df['region_group'] = df['celebrity_homecountry/region'].apply(map_country)

# ==========================================
# 3. 统计与绘图
# ==========================================

# 统计每个大洲的人数
region_counts = df['region_group'].value_counts().reset_index()
region_counts.columns = ['region_group', 'count']

# 按照人数排序 (可选，让图表更清晰)
region_counts = region_counts.sort_values(by='count', ascending=False)

# 绘图
plt.figure(figsize=(12, 7))

# 使用 barplot 绘制柱状图
# 使用自定义颜色，如果类别多于颜色数量，seaborn 会循环使用
ax = sns.barplot(x='region_group', y='count', data=region_counts, palette=custom_palette)

plt.xlabel('Region', fontsize=12)
plt.ylabel('Number of People', fontsize=12)
plt.title(f'Number of People per Region (Season {season_start}-{season_final})', fontsize=14)

# 旋转X轴标签
plt.xticks(rotation=45, ha='right')

# 修改 'North America' 标签为 'North America (excl. US)' 以避免歧义
new_labels = []
for label in ax.get_xticklabels():
    text = label.get_text()
    if text == 'North America':
        new_labels.append('North America (excl. US)')
    else:
        new_labels.append(text)
ax.set_xticklabels(new_labels)

# 添加数值标签
for index, row in region_counts.iterrows():
    plt.text(index, row['count'] + 0.5, str(row['count']), ha='center', va='bottom')

plt.tight_layout()

# 保存
output_filename = 'people-region.png'
save_path = output_filename
if os.path.basename(os.getcwd()) != 'cur代码':
    if not os.path.exists('cur代码'):
        os.makedirs('cur代码')
    save_path = os.path.join('cur代码', output_filename)

plt.savefig(save_path, dpi=300)
plt.close()
print(f"Saved {save_path}")
