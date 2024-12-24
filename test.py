import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# 加载数据
train_data = pd.read_csv('weibo_train_data.txt', sep='\t', header=None,
                         names=['uid', 'mid', 'time', 'forward_count', 'comment_count', 'like_count', 'content'])
predict_data = pd.read_csv('weibo_predict_data.txt', sep='\t', header=None, names=['uid', 'mid', 'time', 'content'])

# 数据预处理
# 例如，将时间转换为年、月、日等特征
train_data['year'] = pd.to_datetime(train_data['time']).dt.year
train_data['month'] = pd.to_datetime(train_data['time']).dt.month
train_data['day'] = pd.to_datetime(train_data['time']).dt.day
predict_data['year'] = pd.to_datetime(predict_data['time']).dt.year
predict_data['month'] = pd.to_datetime(predict_data['time']).dt.month
predict_data['day'] = pd.to_datetime(predict_data['time']).dt.day

# 处理缺失值
train_data['content'].fillna('缺失', inplace=True)
predict_data['content'].fillna('缺失', inplace=True)

# 特征工程
# 使用TF-IDF对博文内容进行向量化
vectorizer = TfidfVectorizer(max_features=500)
X_train_content = vectorizer.fit_transform(train_data['content'])
X_predict_content = vectorizer.transform(predict_data['content'])

# 将用户ID和博文ID转换为数值特征
train_data['uid'] = train_data['uid'].astype('category').cat.codes
train_data['mid'] = train_data['mid'].astype('category').cat.codes
predict_data['uid'] = predict_data['uid'].astype('category').cat.codes
predict_data['mid'] = predict_data['mid'].astype('category').cat.codes

# 合并特征，保持稀疏格式
from scipy.sparse import hstack

X_train = hstack([train_data[['uid', 'mid', 'year', 'month', 'day']].values, X_train_content])
X_predict = hstack([predict_data[['uid', 'mid', 'year', 'month', 'day']].values, X_predict_content])

# 分割训练集和验证集
y_train_forward = train_data['forward_count']
y_train_comment = train_data['comment_count']
y_train_like = train_data['like_count']

# 模型选择与训练
# 这里以随机森林回归器为例
from sklearn.ensemble import RandomForestRegressor

model_forward = RandomForestRegressor(n_estimators=100, random_state=42)
model_comment = RandomForestRegressor(n_estimators=100, random_state=42)
model_like = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
model_forward.fit(X_train, y_train_forward)
model_comment.fit(X_train, y_train_comment)
model_like.fit(X_train, y_train_like)

# 预测
y_predict_forward = model_forward.predict(X_predict)
y_predict_comment = model_comment.predict(X_predict)
y_predict_like = model_like.predict(X_predict)

# 结果处理
# 将预测结果转换为整数
y_predict_forward = np.round(y_predict_forward).astype(int)
y_predict_comment = np.round(y_predict_comment).astype(int)
y_predict_like = np.round(y_predict_like).astype(int)

# 生成提交文件
result = pd.DataFrame({
    'uid': predict_data['uid'],
    'mid': predict_data['mid'],
    'forward_count': y_predict_forward,
    'comment_count': y_predict_comment,
    'like_count': y_predict_like
})

# 格式化结果
result['result'] = result.apply(lambda row: f"{row['forward_count']},{row['comment_count']},{row['like_count']}",
                                axis=1)
result[['uid', 'mid', 'result']].to_csv('weibo_result_data.txt', sep='\t', index=False, header=False)
