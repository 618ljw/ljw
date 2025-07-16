import pandas as pd
import os
from sklearn.preprocessing import OrdinalEncoder
from configuration import FEATURE_SET1, FEATURE_SET2

def load_data(data_path):
    """加载原始数据"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件未找到，请检查路径：{data_path}")
    return pd.read_csv(data_path)

def preprocess_data(df):
    """数据预处理"""
    # 选择有用列并处理缺失值
    df = df[['City Name', 'Variety', 'Date', 'Low Price', 'High Price', 
             'Origin', 'Item Size', 'Package']].dropna()
    
    # 计算平均价格
    df['Average Price'] = (df['Low Price'] + df['High Price']) / 2
    
    # 转换日期格式并提取月份
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    
    return df

def select_features(df, feature_set='set1'):
    """特征选择"""
    if feature_set == 'set1':
        features = FEATURE_SET1
    elif feature_set == 'set2':
        features = FEATURE_SET2
    else:
        raise ValueError("特征集必须是 'set1' 或 'set2'")
    
    # 编码分类特征
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoded_features = encoder.fit_transform(df[features])
    
    # 转换为DataFrame
    encoded_df = pd.DataFrame(
        encoded_features, 
        columns=[f"{col}_encoded" for col in features],
        index=df.index
    )
    
    # 返回特征和目标变量
    X = encoded_df
    y = df['Average Price']
    
    return X, y, encoder, features