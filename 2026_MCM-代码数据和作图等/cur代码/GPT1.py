import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# 1. 数据加载
file_path = r'C:\Users\lenovo\Desktop\2026美赛\2026_MCM-ICM_Problems\cur代码\C_origin.csv'
data = pd.read_csv(file_path)

# 2. 特征选择和预处理
# 假设我们选择裁判的得分作为特征（week1_judge1_score 到 week10_judge4_score）和选手的基本信息（例如年龄、行业等）
# 选择需要的列
judge_score_columns = [col for col in data.columns if 'judge' in col]
selected_columns = ['celebrity_age_during_season', 'celebrity_industry', 'season'] + judge_score_columns
data_selected = data[selected_columns].copy()

# 处理缺失值：填充裁判得分
imputer = SimpleImputer(strategy='mean')
data_selected[judge_score_columns] = imputer.fit_transform(data_selected[judge_score_columns])

# 对类别变量（行业）进行编码
data_selected['celebrity_industry'] = data_selected['celebrity_industry'].astype('category').cat.codes

# 3. 构建目标变量：这里我们根据百分比数据创建排名
# 这里假设你已经有了百分比数据并且需要通过它来预测排名。
# 举个例子：如果是"百分比制"，我们直接取“placement”列作为排名数据
data_selected['placement'] = data['placement']

# 4. 特征和标签分离
X = data_selected.drop(columns=['placement'])
y = data_selected['placement']

# 5. 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 模型训练（以随机森林为例）
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. 预测和评估
y_pred = model.predict(X_test)

# 输出评估结果
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae}')

# 8. 交叉验证
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f'Cross-validated MAE: {np.mean(cv_scores)}')