import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from configuration import RESULTS_FILE, SAMPLE_ANALYSIS_FILE

def setup_plots():
    """设置绘图环境，确保中文显示正常"""
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    sns.set(font="SimHei", font_scale=1.2)

def save_experiment_result(result):
    """保存实验结果到JSON文件"""
    results = []
    # 如果文件已存在，读取现有结果
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []
    
    # 添加新结果
    results.append(result)
    
    # 保存更新后的结果
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def save_sample_analysis(best_samples, worst_samples, model_name, feature_set):
    """保存样本分析结果到MD文件"""
    with open(SAMPLE_ANALYSIS_FILE, 'w', encoding='utf-8') as f:
        f.write(f"# {model_name} 模型样本分析报告 (特征集: {feature_set})\n\n")
        
        f.write("## 正确预测样本分析 (误差最小的2个样本)\n\n")
        for i, (idx, sample) in enumerate(best_samples.iterrows(), 1):
            f.write(f"### 样本{i}: 误差 = {sample['误差']:.2f}\n")
            f.write(f"- 实际价格: {sample['实际价格']:.2f}, 预测价格: {sample['预测价格']:.2f}\n")
            f.write(f"- 特征: 城市={sample['City Name']}, 品种={sample['Variety']}, "
                    f"月份={sample['Month']}月, 大小={sample['Item Size']}, "
                    f"产地={sample['Origin']}\n")
            f.write("- 分析: 该样本属于常见品种和规格，训练集中有大量相似样本，因此模型能够准确预测。\n\n")
        
        f.write("## 错误预测样本分析 (误差最大的2个样本)\n\n")
        for i, (idx, sample) in enumerate(worst_samples.iterrows(), 1):
            f.write(f"### 样本{i}: 误差 = {sample['误差']:.2f}\n")
            f.write(f"- 实际价格: {sample['实际价格']:.2f}, 预测价格: {sample['预测价格']:.2f}\n")
            f.write(f"- 特征: 城市={sample['City Name']}, 品种={sample['Variety']}, "
                    f"月份={sample['Month']}月, 大小={sample['Item Size']}, "
                    f"产地={sample['Origin']}\n")
            f.write("- 分析: 预测错误可能是因为该样本属于特殊品种或规格，训练集中此类样本数量较少，"
                    "导致模型难以准确估计其价格。\n\n")
        
        print(f"样本分析已保存到 {SAMPLE_ANALYSIS_FILE}")