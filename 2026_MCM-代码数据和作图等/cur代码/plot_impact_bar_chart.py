import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
seasons = np.arange(3, 28)
probabilities = [
    0.15, 0.21, 0.30, 0.28, 0.31, 0.26, 0.26, 0.31, 0.23, 0.38,
    0.30, 0.39, 0.30, 0.31, 0.24, 0.36, 0.29, 0.22, 0.16, 0.30,
    0.34, 0.27, 0.32, 0.24, 0.28
]

# 2. 设置颜色
# RGB(145, 172, 224) -> Normalized: (145/255, 172/255, 224/255)
bar_color = (145/255, 172/255, 224/255)
edge_color = 'black'  # 柱子边框颜色

# 3. 创建图表
plt.figure(figsize=(15, 8))
ax = plt.gca()

# 设置背景色
ax.set_facecolor('#f0f0f0')  # 浅灰色背景

# 绘制柱状图
bars = plt.bar(seasons, probabilities, color=bar_color, edgecolor=edge_color, width=0.8, linewidth=0.5)

# 4. 添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{height:.2f}',
             ha='center', va='bottom', fontsize=10)

# 5. 设置标题和标签
plt.title('Average Impact of Ranking System vs Percentage System by Season (S3-S27)\n(Height = Probability of Different Outcome)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Season', fontsize=14)
plt.ylabel('Probability of Outcome Change', fontsize=14)

# 6. 设置坐标轴刻度和范围
plt.xticks(seasons, fontsize=12)
plt.ylim(0, 0.45)  # 根据数据范围适当调整上限

# 7. 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7, color='#bdbdbd')
plt.grid(axis='x', linestyle='--', alpha=0.5, color='#bdbdbd')
ax.set_axisbelow(True)  # 让网格线在柱子下方

# 移除上边框和右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# 也可以像原图那样保留边框但设为灰色
for spine in ax.spines.values():
    spine.set_color('#cccccc')

# 8. 保存图片
output_path = 'impact_bar_chart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图表已生成并保存至: {output_path}")
