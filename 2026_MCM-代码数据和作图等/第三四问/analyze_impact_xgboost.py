import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import re

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义颜色
rank1_color = (77/255, 103/255, 164/255)    # #4D67A4
rank3_color = (186/255, 168/255, 210/255)  # #BAA8D2
rank5_color = (255/255, 204/255, 154/255)  # #FFCC9A
rank7_color = (189/255, 115/255, 106/255)  # #BD736A
rank8_color = (121/255, 77/255, 72/255)    # #794D48

def plot_custom_shap_bar(shap_values, title, ax):
    # 计算平均绝对SHAP值
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    feature_names = shap_values.feature_names
    
    # 排序 (从小到大，因为barh是从下往上画)
    sorted_idx = np.argsort(mean_shap)
    sorted_mean_shap = mean_shap[sorted_idx]
    sorted_features = [feature_names[i] for i in sorted_idx]
    
    # 颜色分配：最重要的（最上面）用 rank1_color，依此类推
    # 因为barh是从下往上画，所以最后一个（最重要的）对应 rank1_color
    colors_list = [rank1_color, rank3_color, rank5_color, rank7_color]
    # 截取需要的颜色数量
    current_colors = colors_list[:len(sorted_features)]
    # 倒序，使得最重要的特征获得列表中的第一个颜色
    bar_colors = list(reversed(current_colors))
    
    # 绘制水平柱状图
    bars = ax.barh(range(len(sorted_features)), sorted_mean_shap, color=bar_colors)
    
    # 设置标签
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("mean(|SHAP value|)", fontsize=12)
    
    # 去除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 添加数值标签
    max_val = sorted_mean_shap.max()
    for i, v in enumerate(sorted_mean_shap):
        ax.text(v + max_val * 0.02, i, f"+{v:.2f}", color=rank8_color, va='center', fontsize=12, fontweight='bold')

def analyze_impact():
    # 1. 数据读取与预处理
    print("正在读取数据...")
    # 读取原始数据 (C_origin.csv) - 假设在当前目录下
    df_origin = pd.read_csv('C_origin.csv')
    df_origin.rename(columns={'celebrity_name': 'celebrity'}, inplace=True)
    
    # 过滤掉 C_origin 中没有粉丝数据的早期赛季 (Season 1-2)
    df_origin = df_origin[df_origin['season'] >= 3]
    
    # 读取粉丝投票数据
    df_fan = pd.read_csv('粉丝投票结果-百分比-3-27.csv')
    
    # 2. 构建特征矩阵
    feature_cols = [
        'celebrity', 'season', 'celebrity_age_during_season', 
        'celebrity_industry', 'celebrity_homecountry/region', 'ballroom_partner'
    ]
    target_cols = ['placement']
    
    df_model = df_origin[feature_cols + target_cols].copy()
    
    # 处理粉丝数据
    week_cols_fan = [col for col in df_fan.columns if col.startswith('week')]
    df_fan['avg_fan_vote'] = df_fan[week_cols_fan].mean(axis=1)
    
    # 处理裁判数据：从 C_origin 计算
    week_cols_origin = [f'week{i}_judge{j}_score' for i in range(1, 12) for j in range(1, 5)]
    week_cols_origin = [col for col in week_cols_origin if col in df_origin.columns]
    
    for col in week_cols_origin:
        df_origin[col] = pd.to_numeric(df_origin[col], errors='coerce')
        
    df_origin['avg_judge_score'] = df_origin[week_cols_origin].mean(axis=1)
    
    # 更新 df_model
    df_model = df_origin.copy()
    
    # 清理数据
    df_model['celebrity'] = df_model['celebrity'].astype(str).str.strip()
    df_fan['celebrity'] = df_fan['celebrity'].astype(str).str.strip()
    
    def clean_name(name):
        return re.sub(r'[^a-zA-Z0-9\s]', '', name).lower().strip()
    
    df_model['celebrity_clean'] = df_model['celebrity'].apply(clean_name)
    df_fan['celebrity_clean'] = df_fan['celebrity'].apply(clean_name)
    
    # 合并 Fan 数据
    df_merged = df_model.merge(df_fan[['celebrity_clean', 'season', 'avg_fan_vote']], on=['celebrity_clean', 'season'], how='inner')
    df_model = df_merged
    
    # 去除缺失值
    df_model.dropna(subset=['avg_fan_vote', 'avg_judge_score'], inplace=True)
    print(f"有效样本数: {len(df_model)}")
    
    if len(df_model) == 0:
        print("错误：合并后样本数为0。")
        return

    # 3. 特征编码
    # 3.2 Ballroom Partner: Target Encoding
    partner_stats = df_model.groupby('ballroom_partner')['placement'].mean().reset_index()
    partner_stats.rename(columns={'placement': 'partner_strength'}, inplace=True)
    df_model = df_model.merge(partner_stats, on='ballroom_partner', how='left')
    
    # 3.3 Industry & Region: Label Encoding
    le_industry = LabelEncoder()
    df_model['industry_code'] = le_industry.fit_transform(df_model['celebrity_industry'].astype(str))
    
    le_region = LabelEncoder()
    df_model['region_code'] = le_region.fit_transform(df_model['celebrity_homecountry/region'].astype(str))
    
    # 最终特征列表
    X_features = ['celebrity_age_during_season', 'industry_code', 'region_code', 'partner_strength']
    feature_names = ['Age', 'Industry', 'Region', 'Pro Dancers']
    
    X = df_model[X_features]
    
    # 定义三个目标变量
    y_placement = df_model['placement']
    y_judge = df_model['avg_judge_score']
    y_fan = df_model['avg_fan_vote']
    
    # 4. 模型训练与 SHAP 分析
    def train_and_explain(X, y, model_name):
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=4, learning_rate=0.1)
        model.fit(X, y)
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        shap_values.feature_names = feature_names
        return model, shap_values

    print("正在训练模型 A (Placement)...")
    model_placement, shap_placement = train_and_explain(X, y_placement, "Placement")
    
    print("正在训练模型 B (Judge Score)...")
    model_judge, shap_judge = train_and_explain(X, y_judge, "Judge Score")
    
    print("正在训练模型 C (Fan Vote)...")
    model_fan, shap_fan = train_and_explain(X, y_fan, "Fan Vote")
    
    # 5. 可视化结果
    
    # 5.1 总体特征重要性对比 (Bar Plot) - 使用自定义绘图
    # 增加宽度: (22, 6)
    plt.figure(figsize=(15, 4))
    
    ax1 = plt.subplot(1, 3, 1)
    plot_custom_shap_bar(shap_placement, "Impact on Final Placement\n(Lower Placement is Better)", ax1)
    
    ax2 = plt.subplot(1, 3, 2)
    plot_custom_shap_bar(shap_judge, "Impact on Judge Scores", ax2)
    
    ax3 = plt.subplot(1, 3, 3)
    plot_custom_shap_bar(shap_fan, "Impact on Fan Votes", ax3)
    
    plt.tight_layout()
    plt.savefig('shap_importance_comparison.png', dpi=300)
    plt.close()
    
    # 5.2 蜂群图 (Beeswarm Plot)
    # 增加宽度: (22, 6)
    plt.figure(figsize=(22, 6))
    
    plt.subplot(1, 2, 1)
    shap.plots.beeswarm(shap_judge, show=False)
    plt.title("How Features Impact Judge Scores")
    
    plt.subplot(1, 2, 2)
    shap.plots.beeswarm(shap_fan, show=False)
    plt.title("How Features Impact Fan Votes")
    
    plt.tight_layout()
    plt.savefig('shap_beeswarm_comparison.png', dpi=300)
    plt.close()
    
    # 5.3 依赖图 (Dependence Plot)
    # 增加宽度
    plt.figure(figsize=(16, 5))
    
    plt.subplot(1, 2, 1)
    shap.plots.scatter(shap_judge[:, "Age"], color=shap_judge, show=False)
    plt.title("Age vs Judge Impact")
    plt.ylabel("SHAP value for Judge Score")
    
    plt.subplot(1, 2, 2)
    shap.plots.scatter(shap_fan[:, "Age"], color=shap_fan, show=False)
    plt.title("Age vs Fan Impact")
    plt.ylabel("SHAP value for Fan Vote")
    
    plt.tight_layout()
    plt.savefig('shap_dependence_age.png', dpi=300)
    plt.close()

    print("所有分析完成，图片已保存至当前目录。")
    print("- shap_importance_comparison.png: 特征重要性对比 (自定义配色)")
    print("- shap_beeswarm_comparison.png: 影响方向对比 (加宽)")
    print("- shap_dependence_age.png: 年龄的具体影响趋势")

if __name__ == "__main__":
    analyze_impact()
