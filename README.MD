# 鸢尾花分类对比实验 (Iris Classification)

基于 Scikit-learn 对鸢尾花数据集进行 3 种经典分类模型的训练、调参与对比。

## 📁 项目结构

```
├── iris_main.py            # 主程序：数据探索 + 模型训练 + 评估
├── iris_model_knn.pkl      # 保存的 KNN 模型
├── iris_model_tree.pkl     # 保存的决策树模型
├── iris_model_linear.pkl   # 保存的逻辑回归模型
└── README.md
```

## 🚀 快速开始

### 环境要求
- Python 3.x
- 依赖库：`joblib`, `matplotlib`, `scikit-learn`, `pandas`, `seaborn`, `numpy`

安装依赖：
```bash
pip install joblib matplotlib scikit-learn pandas seaborn numpy
```

### 运行
```bash
python iris_main.py
```

## 📊 主要功能

1. **数据探索**：散点图、箱线图、特征相关性热力图
2. **模型训练与对比**：
   - KNN（网格搜索最优 k 值）
   - 决策树（网格搜索最优 max_depth）
   - 逻辑回归
3. **评估指标**：准确率、混淆矩阵、分类报告、5折交叉验证
4. **模型保存**：训练完成后自动保存为 `.pkl` 文件

## 📈 实验结果（示例）

| 模型 | 测试集准确率 | 交叉验证平均准确率 |
|------|-------------|-------------------|
| KNN | 1.0 | ~0.96 |
| 决策树 | 1.0 | ~0.95 |
| 逻辑回归 | 1.0 | ~0.97 |

> 注：鸢尾花数据集较简单，3个模型均能达到很高准确率。

## 📝 备注

- 数据集：`sklearn.datasets.load_iris()`（内置）
- 随机种子：`random_state=42`（保证可复现）