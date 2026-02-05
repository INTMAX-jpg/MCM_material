import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
# Season 1-27
seasons = np.arange(1, 28)

# Dataset 1: Fan percentage distance
fan_data = [
    0.88, 0.73, 0.90, 0.70, 0.42, 0.77, 0.97, 0.63, 0.73,
    0.60, 0.80, 0.99, 0.30, 0.72, 0.82, 0.79, 0.55, 0.71,
    0.95, 0.76, 0.68, 1.00, 0.66, 0.75, 0.85, 0.74, 0.92
]

# Dataset 2: Judge percentage distance
judge_data = [
    0.94, 0.81, 0.70, 1.00, 0.85, 0.91, 0.88, 0.30, 0.83,
    0.97, 0.80, 0.90, 0.87, 0.92, 0.73, 0.82, 0.95, 0.76,
    0.84, 0.93, 0.86, 0.65, 0.89, 0.98, 0.81, 0.78, 0.96
]

# 颜色定义
rank3_color = (186/255, 168/255, 210/255)  # #BAA8D2
rank5_color = (255/255, 204/255, 154/255)  # #FFCC9A

# 2. 绘制图表 (折线图)
plt.figure(figsize=(12, 6))

# 绘制 Fan 折线
plt.plot(seasons, fan_data, color=rank3_color, label=r'$dis_{\text{Fan,percentage}}$', linewidth=2.5, marker='o')

# 绘制 Judge 折线
plt.plot(seasons, judge_data, color=rank5_color, label=r'$dis_{\text{Judge,percentage}}$', linewidth=2.5, marker='o')

# 3. 图表装饰
plt.xlabel('season', fontsize=12)
plt.ylabel('avg_dis', fontsize=12)
plt.title(r'$dis$ Distribution for Percentage System', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(np.arange(1, 28, 2))  # 每2个赛季显示一个刻度

# 4. 保存图片
output_path = 'avg_dis_curves.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图表已生成并保存至: {output_path}")
