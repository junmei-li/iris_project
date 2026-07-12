import joblib
import matplotlib.pyplot as plt
from matplotlib.pyplot import title
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import seaborn as sns

iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['target'] = iris.target
X = iris.data #特征
y = iris.target #标签

# 1. 查看数据集基本信息
print("数据集形状（样本数 × 特征数）：", iris.data.shape)
print("特征名称：", iris.feature_names)
print("类别名称：", iris.target_names)
print("标签取值：", set(iris.target))  # 0:山鸢尾, 1:变色鸢尾, 2:维吉尼亚鸢尾

# 2. 查看前5条样本，直观感受数据
print("\n前5条样本特征：\n", iris.data[:5])
print("前5条样本标签：", iris.target[:5])

# 3. 统计每个类别的样本数量（验证数据均衡）
import numpy as np
for i in range(3):
    print(f"\n类别 {iris.target_names[i]} 的样本数：", np.sum(iris.target == i))

#训练之前：探索数据
plt.figure(figsize=(8,6))
#散点图
scatter = plt.scatter(df["petal length (cm)"],df["petal width (cm)"],c = df['target'],cmap='viridis',s = 60)
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.title('Petal length vs Petal width (By Species)' )
plt.legend(*scatter.legend_elements(),title = 'Species')
plt.show()

#箱线图
plt.figure(figsize=(10,6))
sns.boxplot(x = 'target',y = 'petal length (cm)',data=df,palette='Set2')
plt.xlabel('Species')
plt.ylabel('Petal length (cm)')
plt.title('Petal Length Distribution by Species')
plt.show()

#热力图
plt.figure(figsize=(8,6))
corr = df.corr()
sns.heatmap(corr,annot=True,cmap='coolwarm',fmt='.2f',linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()




















#划分数据
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train) 后面创建pipeline里面已经有了scaler所以在这里不用再缩放一遍

#训练模型knn
# model_knn = KNeighborsClassifier(n_neighbors=3)
# model_knn.fit(X_train,y_train) 直接利用pipeline

#创建pipeline（knn
best_k = 0
best_score = 0
k_range=range(1,21)
k_score = []
for i in range(1,21):
    pipeline = Pipeline([("scaler",StandardScaler()),
                         ("knn",KNeighborsClassifier(n_neighbors=i))])
    #pipeline.fit(X_train,y_train) 下面cross会自动fit
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    mean_score = scores.mean()
    k_score.append(mean_score)
    if(mean_score > best_score):
        best_k = i
        best_score = mean_score
k_range=range(1,21)
plt.plot(k_range,k_score)
plt.show()
pipeline = Pipeline([("scaler",StandardScaler()),
                         ("knn",KNeighborsClassifier(n_neighbors=best_k))])
pipeline.fit(X_train,y_train)
#knn的模型评估
y_pred_knn = pipeline.predict(X_test)
acc_knn = accuracy_score(y_test,y_pred_knn )#准确率
cm_knn = confusion_matrix(y_test,y_pred_knn )#混淆矩阵
cr_knn = classification_report(y_test,y_pred_knn )#分类报告
scores_knn = cross_val_score(pipeline,X,y,cv=5)#交叉验证
print("准确率：",acc_knn)
print(cm_knn)
print(cr_knn)
print("5折交叉验证结果：",scores_knn)
print("平均准确率：",scores_knn.mean())
joblib.dump(pipeline,"iris_model_knn.pkl")


#创建pipeline（决策树
best_depth = 0
best_score = 0
depth_score = []
for i in range(1,21):
    pipeline = Pipeline([("scaler",StandardScaler()),
                         ("tree",DecisionTreeClassifier(max_depth=i))])
    #pipeline.fit(X_train,y_train) 下面cross会自动fit
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    mean_score = scores.mean()
    depth_score.append(mean_score)
    if(mean_score > best_score):
        best_depth = i
        best_score = mean_score

depth_range = range(1,21)
plt.plot(depth_range,depth_score)
plt.show()

pipeline = Pipeline([("scaler",StandardScaler()),
                         ("tree",DecisionTreeClassifier(max_depth=best_depth))])
pipeline.fit(X_train,y_train)
#tree的模型评估
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test,y_pred)#准确率
cm = confusion_matrix(y_test,y_pred)#混淆矩阵
cr = classification_report(y_test,y_pred)#分类报告
scores_tree = cross_val_score(pipeline,X,y,cv=5)#交叉验证
print("准确率：",acc)
print(cm)
print(cr)
print("5折交叉验证结果：",scores_tree)
print("平均准确率：",scores_tree.mean())
joblib.dump(pipeline,"iris_model_tree.pkl")

#逻辑回归
pipeline = Pipeline([("scaler",StandardScaler()),("linear",LogisticRegression(C = 1.0,max_iter=1000))])
pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test,y_pred)#准确率
cm = confusion_matrix(y_test,y_pred)#混淆矩阵
cr = classification_report(y_test,y_pred)#分类报告
scores_linear = cross_val_score(pipeline,X,y,cv=5)#交叉验证
print("准确率：",acc)
print(cm)
print(cr)
print("5折交叉验证结果：",scores_linear)
print("平均准确率：",scores_linear.mean())
joblib.dump(pipeline,"iris_model_linear.pkl")

