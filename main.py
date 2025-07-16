from configuration import DATA_PATH, CV_FOLDS, EXPERIMENTS
from utility import setup_plots, save_experiment_result, save_sample_analysis
from feature_processing import load_data, preprocess_data
from data_analysis import (
    plot_price_distribution, plot_variety_price, 
    plot_city_price, plot_monthly_trend, plot_size_price
)
from evaluate import print_data_summary, evaluate_correlation, cross_validate_model, analyze_samples
from model import train_linear_regression, train_lgbm, train_xgboost

def run_experiment(df, model_name, params, feature_set):
    """运行单个实验并保存结果"""
    # 选择模型训练函数
    if model_name == 'LR':
        model_trainer = lambda X, y: train_linear_regression(X, y, params)
    elif model_name == 'LGBM':
        model_trainer = lambda X, y: train_lgbm(X, y, params)
    elif model_name == 'XGBoost':
        model_trainer = lambda X, y: train_xgboost(X, y, params)
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    # 执行交叉验证
    results = cross_validate_model(df, model_trainer, feature_set, CV_FOLDS)
    
    # 添加编码信息
    results['fea_encoding'] = 'ordinal'
    
    # 保存实验结果
    save_experiment_result(results)
    
    return results

def run_sample_analysis(df, model_name, params, feature_set):
    """运行样本分析并保存结果"""
    # 选择模型训练函数
    if model_name == 'LR':
        model_trainer = lambda X, y: train_linear_regression(X, y, params)
    elif model_name == 'LGBM':
        model_trainer = lambda X, y: train_lgbm(X, y, params)
    elif model_name == 'XGBoost':
        model_trainer = lambda X, y: train_xgboost(X, y, params)
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    # 特征选择
    from feature_processing import select_features
    X, y, encoder, features = select_features(df, feature_set)
    
    # 训练模型
    model, _, _ = model_trainer(X, y)
    
    # 样本分析
    best_samples, worst_samples = analyze_samples(df, model, feature_set)
    
    # 保存样本分析结果
    save_sample_analysis(best_samples, worst_samples, model_name, feature_set)
    
    return best_samples, worst_samples

def main():
    """主程序入口函数"""
    # 设置绘图环境
    setup_plots()
    
    # 加载和预处理数据
    df = load_data(DATA_PATH)
    processed_df = preprocess_data(df)
    
    # 数据探索性分析
    print_data_summary(processed_df)
    evaluate_correlation(processed_df)
    
    # 数据可视化
    plot_price_distribution(processed_df)
    plot_variety_price(processed_df)
    plot_city_price(processed_df)
    plot_monthly_trend(processed_df)
    plot_size_price(processed_df)
    
    # 运行所有实验
    print("\n开始模型训练和评估...")
    for experiment in EXPERIMENTS:
        print(f"\n运行实验: {experiment['model']} + 特征集{experiment['feature_set'][-1]}")
        results = run_experiment(processed_df, experiment['model'], 
                                experiment['params'], experiment['feature_set'])
        print(f"实验完成，平均测试RMSE: {results['average_test_performance']['rmse']}")
    
    # 运行样本分析 - 使用LGBM模型和特征集2
    print("\n开始样本分析...")
    lgbm_experiment = next(exp for exp in EXPERIMENTS 
                          if exp['model'] == 'LGBM' and exp['feature_set'] == 'set2')
    best_samples, worst_samples = run_sample_analysis(
        processed_df, lgbm_experiment['model'], 
        lgbm_experiment['params'], lgbm_experiment['feature_set']
    )
    
    print("\n所有实验完成，结果已保存到output目录")
    print("样本分析完成，分析报告已保存到output目录")

if __name__ == '__main__':
    main()