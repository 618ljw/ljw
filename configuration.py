import os

# 路径配置
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'US-pumpkins.csv')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
IMAGES_DIR = os.path.join(ROOT_DIR, 'output', 'images')
RESULTS_FILE = os.path.join(OUTPUT_DIR, 'experiment_results.json')
SAMPLE_ANALYSIS_FILE = os.path.join(OUTPUT_DIR, 'sample_analysis.md')

# 确保目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# 模型参数配置
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'n_estimators': 100,
    'random_state': 42
}

XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.05,
    'n_estimators': 100,
    'random_state': 42
}

# 交叉验证配置
CV_FOLDS = 3
RANDOM_STATE = 42

# 特征配置 - 两组不同的特征选择
FEATURE_SET1 = ['Variety', 'Origin', 'Item Size', 'Month', 'Package']
FEATURE_SET2 = ['Variety', 'City Name', 'Origin', 'Item Size', 'Month', 'Package', 'City Name']

TARGET_COLUMN = 'Average Price'

# 实验配置
EXPERIMENTS = [
    {'model': 'LR', 'params': None, 'feature_set': 'set1', 'encoding': 'ordinal'},
    {'model': 'LR', 'params': None, 'feature_set': 'set2', 'encoding': 'ordinal'},
    {'model': 'LGBM', 'params': LGBM_PARAMS, 'feature_set': 'set1', 'encoding': 'ordinal'},
    {'model': 'LGBM', 'params': LGBM_PARAMS, 'feature_set': 'set2', 'encoding': 'ordinal'},
    {'model': 'XGBoost', 'params': XGBOOST_PARAMS, 'feature_set': 'set1', 'encoding': 'ordinal'},
    {'model': 'XGBoost', 'params': XGBOOST_PARAMS, 'feature_set': 'set2', 'encoding': 'ordinal'}
]