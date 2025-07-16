import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LinearRegression

def train_linear_regression(X_train, y_train, params=None):
    """训练线性回归模型"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, "LR", params

def train_lgbm(X_train, y_train, params=None):
    """训练LGBM模型"""
    if params is None:
        from configuration import LGBM_PARAMS
        params = LGBM_PARAMS
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model, "LGBM", params

def train_xgboost(X_train, y_train, params=None):
    """训练XGBoost模型"""
    if params is None:
        from configuration import XGBOOST_PARAMS
        params = XGBOOST_PARAMS
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model, "XGBoost", params

def predict_model(model, X):
    """使用模型进行预测"""
    return model.predict(X)