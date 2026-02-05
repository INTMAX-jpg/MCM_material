import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def analyze_impact():
    # 1. 数据读取与预处理
    print("正在读取数据...")
    # 读取原始数据 (C_origin.csv)
    # 注意：这里我们使用原始数据中的特征
    df_origin = pd.read_csv(r'c:\Users\lenovo\Desktop\2026美赛\2026_MCM-ICM_Problems\cur代码\C_origin.csv')
    df_origin.rename(columns={'celebrity_name': 'celebrity'}, inplace=True) # 重命名列以匹配
    
    # 过滤掉 C_origin 中没有粉丝数据的早期赛季 (Season 1-2)
    # 根据 Fan Data 示例，数据是从 Season 3 开始的
    df_origin = df_origin[df_origin['season'] >= 3]
    
    # 读取粉丝投票数据 (模型预测-反向求百分比结果.csv)
    # 假设这个文件包含了我们需要的粉丝投票数据
    df_fan = pd.read_csv(r'c:\Users\lenovo\Desktop\2026美赛\2026_MCM-ICM_Problems\cur代码\模型预测-反向求百分比结果.csv')
    
    # 读取裁判评分数据 (judge排名制-每周平均分数(十分制)&排名.csv)
    df_judge = pd.read_csv(r'c:\Users\lenovo\Desktop\2026美赛\2026_MCM-ICM_Problems\cur代码\judge排名制-每周平均分数(十分制)&排名.csv')

    # 2. 构建特征矩阵 (Feature Engineering)
    # 我们以 C_origin 为基础，因为它包含所有静态特征 (Age, Industry, etc.)
    # 选取关键特征
    feature_cols = [
        'celebrity', 'season', 'celebrity_age_during_season', 
        'celebrity_industry', 'celebrity_homecountry/region', 'ballroom_partner'
    ]
    target_cols = ['placement']
    
    df_model = df_origin[feature_cols + target_cols].copy()
    
    # 处理粉丝数据：计算每个选手的平均粉丝得票率
    # 模型预测-反向求百分比结果.csv 中的列名是 week1, week2 等
    week_cols_fan = [col for col in df_fan.columns if col.startswith('week')]
    df_fan['avg_fan_vote'] = df_fan[week_cols_fan].mean(axis=1)
    
    # 处理裁判数据：计算每个选手的平均裁判得分
    # 调试发现 judge 数据只包含 Season 1, 2, 28-34，缺少 Season 3-27 的大部分数据
    # 因此我们尝试直接从 C_origin 计算裁判平均分
    week_cols_origin = [f'week{i}_judge{j}_score' for i in range(1, 12) for j in range(1, 5)]
    # 确保列存在
    week_cols_origin = [col for col in week_cols_origin if col in df_origin.columns]
    
    # C_origin 中的分数包含 'N/A'，需要处理
    # 将 'N/A' 替换为 NaN，并转换为数值
    for col in week_cols_origin:
        df_origin[col] = pd.to_numeric(df_origin[col], errors='coerce')
        
    df_origin['avg_judge_score'] = df_origin[week_cols_origin].mean(axis=1)
    
    # 更新 df_model
    df_model = df_origin.copy()
    
    # 清理数据：去除字符串两端的空格
    df_model['celebrity'] = df_model['celebrity'].astype(str).str.strip()
    df_fan['celebrity'] = df_fan['celebrity'].astype(str).str.strip()
    
    # 特殊处理：检查并修复名字不一致
    import re
    def clean_name(name):
        # 只保留字母、数字和空格，去除特殊符号
        return re.sub(r'[^a-zA-Z0-9\s]', '', name).lower().strip()
    
    df_model['celebrity_clean'] = df_model['celebrity'].apply(clean_name)
    df_fan['celebrity_clean'] = df_fan['celebrity'].apply(clean_name)
    
    print("Cleaned name examples:")
    print("Model:", df_model['celebrity_clean'].head().tolist())
    print("Fan:", df_fan['celebrity_clean'].head().tolist())
    
    # 只需要合并 Fan 数据，Judge 数据已经直接计算
    df_merged = df_model.merge(df_fan[['celebrity_clean', 'season', 'avg_fan_vote']], on=['celebrity_clean', 'season'], how='inner')
    print(f"After Fan merge (clean): {len(df_merged)}")
    
    df_model = df_merged
    
    # 去除缺失值 (某些选手可能没有粉丝数据或裁判数据)
    df_model.dropna(subset=['avg_fan_vote', 'avg_judge_score'], inplace=True)
    
    print(f"有效样本数: {len(df_model)}")
    if len(df_model) == 0:
        print("错误：合并后样本数为0。请检查数据源的 'celebrity' 和 'season' 列是否匹配。")
        # 调试信息
        print("C_origin 示例:", df_origin[['celebrity', 'season']].head())
        print("Fan Data 示例:", df_fan[['celebrity', 'season']].head())
        return

    # 3. 特征编码
    # 3.1 Industry & Region: One-Hot Encoding or Label Encoding? 
    # 为了 XGBoost 方便，且类别数不多，可以用 Label Encoding，或者 XGBoost 自带的类别处理
    # 这里我们手动处理一些高基数特征
    
    # 3.2 Ballroom Partner: Target Encoding
    # 舞伴人数较多 (60+)，直接 One-Hot 会太稀疏。
    # 我们用“该舞伴历史选手的平均排名”作为特征值，代表舞伴的实力
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
    feature_names = ['Age', 'Industry', 'Region', 'Partner Strength']
    
    X = df_model[X_features]
    
    # 定义三个目标变量
    y_placement = df_model['placement']
    y_judge = df_model['avg_judge_score']
    y_fan = df_model['avg_fan_vote']
    
    # 4. 模型训练与 SHAP 分析
    
    # 通用训练函数
    def train_and_explain(X, y, model_name):
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=4, learning_rate=0.1)
        model.fit(X, y)
        
        # 计算 SHAP 值
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        shap_values.feature_names = feature_names # 重命名特征以便显示
        
        return model, shap_values

    print("正在训练模型 A (Placement)...")
    model_placement, shap_placement = train_and_explain(X, y_placement, "Placement")
    
    print("正在训练模型 B (Judge Score)...")
    model_judge, shap_judge = train_and_explain(X, y_judge, "Judge Score")
    
    print("正在训练模型 C (Fan Vote)...")
    model_fan, shap_fan = train_and_explain(X, y_fan, "Fan Vote")
    
    # 5. 可视化结果
    
    # 5.1 总体特征重要性对比 (Bar Plot)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    shap.plots.bar(shap_placement, show=False, max_display=10)
    plt.title("Impact on Final Placement\n(Lower Placement is Better)")
    
    plt.subplot(1, 3, 2)
    shap.plots.bar(shap_judge, show=False, max_display=10)
    plt.title("Impact on Judge Scores")
    
    plt.subplot(1, 3, 3)
    shap.plots.bar(shap_fan, show=False, max_display=10)
    plt.title("Impact on Fan Votes")
    
    plt.tight_layout()
    plt.savefig('shap_importance_comparison.png', dpi=300)
    plt.close()
    
    # 5.2 蜂群图 (Beeswarm Plot) - 展示影响方向
    # 对比 Judge 和 Fan
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    shap.plots.beeswarm(shap_judge, show=False)
    plt.title("How Features Impact Judge Scores")
    
    plt.subplot(1, 2, 2)
    shap.plots.beeswarm(shap_fan, show=False)
    plt.title("How Features Impact Fan Votes")
    
    plt.tight_layout()
    plt.savefig('shap_beeswarm_comparison.png', dpi=300)
    plt.close()
    
    # 5.3 依赖图 (Dependence Plot) - Age 的影响
    # 观察 Age 对 Judge 和 Fan 的非线性影响
    plt.figure(figsize=(12, 5))
    
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
    print("- shap_importance_comparison.png: 特征重要性对比")
    print("- shap_beeswarm_comparison.png: 影响方向对比")
    print("- shap_dependence_age.png: 年龄的具体影响趋势")

if __name__ == "__main__":
    analyze_impact()
