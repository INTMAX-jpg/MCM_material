import matplotlib.pyplot as plt

# 1. 准备数据
categories = ['Inconsistent', 'Consistent']
values = [69, 166]

# 2. 设置颜色
rank3_color = (186/255, 168/255, 210/255)  # #BAA8D2 (Inconsistent)
rank5_color = (255/255, 204/255, 154/255)  # #FFCC9A (Consistent)
colors = [rank3_color, rank5_color]

# 3. 创建图表
plt.figure(figsize=(8, 6))
bars = plt.bar(categories, values, color=colors, width=0.6)

# 4. 添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# 5. 设置标题和标签
plt.title('Consistency Analysis of the Elimination Status of Two Systems', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Num of weeks', fontsize=12)

# 6. 设置网格线 (仅 Y 轴)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.gca().set_axisbelow(True)

# 7. 保存图片
output_path = 'consistency_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图表已生成并保存至: {output_path}")
