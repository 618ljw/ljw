import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from configuration import IMAGES_DIR
import os

def print_data_summary(df):
    """打印数据摘要信息"""
    print("数据摘要信息:")
    print(f"样本数量: {df.shape[0]}")
    print(f"特征数量: {df.shape[1] - 1}")  # 减去目标变量
    print(f"平均价格: {df['Average Price'].mean():.2f}")
    print(f"价格标准差: {df['Average Price'].std():.2f}")
    print(f"价格范围: [{df['Average Price'].min():.2f}, {df['Average Price'].max():.2f}]")

def evaluate_correlation(df):
    """评估特征相关性"""
    # 选择数值特征计算相关性
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    
    # 绘制相关性热图
    plt.figure(figsize=(10, 8))
    plt.title('特征相关性热图')
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'correlation_heatmap.png'))
    plt.close()
    
    return corr_matrix

def evaluate_model(y_true, y_pred):
    """评估模型性能"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'rmse': f"{rmse:.2f}",
        'mae': f"{mae:.2f}",
        'r2': f"{r2:.2f}"
    }

def cross_validate_model(df, model_trainer, feature_set='set1', cv_folds=3):
    """交叉验证模型"""
    from sklearn.model_selection import KFold
    from feature_processing import select_features
    
    # 特征选择
    X, y, encoder, features = select_features(df, feature_set)
    
    # 初始化K折交叉验证
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # 存储每折的结果
    results = {
        'feature_set': feature_set,
        'features': features,
        'fold_results': []
    }
    
    # 进行交叉验证
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # 训练模型
        model, model_name, model_params = model_trainer(X_train, y_train)
        
        # 预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # 评估
        train_metrics = evaluate_model(y_train, y_train_pred)
        test_metrics = evaluate_model(y_test, y_test_pred)
        
        # 保存结果
        fold_result = {
            f'{fold}_fold_train_data': [X_train.shape[0], X_train.shape[1]],
            f'{fold}_fold_test_data': [X_test.shape[0], X_test.shape[1]],
            f'{fold}_fold_train_performance': train_metrics,
            f'{fold}_fold_test_performance': test_metrics
        }
        results['fold_results'].append(fold_result)
    
    # 计算平均性能
    results['model_name'] = model_name
    results['model_params'] = model_params
    results['average_train_performance'] = calculate_average_metrics(
        results['fold_results'], 'train_performance')
    results['average_test_performance'] = calculate_average_metrics(
        results['fold_results'], 'test_performance')
    
    return results

def calculate_average_metrics(fold_results, metric_type):
    """计算多折的平均性能指标"""
    metrics = ['rmse', 'mae', 'r2']
    avg_metrics = {}
    
    for metric in metrics:
        values = []
        for fold in fold_results:
            fold_key = list(fold.keys())[list(fold.keys()).index(
                next(k for k in fold.keys() if metric_type in k))]
            values.append(float(fold[fold_key][metric]))
        avg_metrics[metric] = f"{sum(values) / len(values):.2f}"
    
    return avg_metrics

def analyze_samples(df, model, feature_set='set1', n_samples=2):
    """分析样本预测结果"""
    from feature_processing import select_features
    
    X, y, encoder, features = select_features(df, feature_set)
    
    # 预测所有样本
    y_pred = model.predict(X)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        '实际价格': y,
        '预测价格': y_pred,
        '误差': abs(y - y_pred)
    })
    
    # 合并原始特征以便分析
    results_df = pd.concat([results_df, df.reset_index(drop=True)], axis=1)
    
    # 找出预测最好和最差的样本
    best_samples = results_df.nsmallest(n_samples, '误差')
    worst_samples = results_df.nlargest(n_samples, '误差')
    
    return best_samples, worst_samples