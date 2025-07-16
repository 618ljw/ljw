import matplotlib.pyplot as plt
import seaborn as sns
import os
from configuration import IMAGES_DIR

def plot_price_distribution(df):
    """绘制价格分布直方图"""
    plt.figure(figsize=(10, 6))
    plt.hist(df['Average Price'], bins=30, color='orange', alpha=0.7)
    plt.title('南瓜价格分布')
    plt.xlabel('平均价格')
    plt.ylabel('频数')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(IMAGES_DIR, 'price_distribution.png'))
    plt.close()

def plot_variety_price(df):
    """绘制不同品种价格箱线图"""
    plt.figure(figsize=(12, 8))
    top_varieties = df['Variety'].value_counts().head(5).index
    sns.boxplot(x='Variety', y='Average Price', data=df[df['Variety'].isin(top_varieties)])
    plt.title('不同南瓜品种的价格分布')
    plt.xlabel('品种')
    plt.ylabel('平均价格')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'variety_price.png'))
    plt.close()

def plot_city_price(df):
    """绘制不同城市价格条形图"""
    plt.figure(figsize=(12, 8))
    top_cities = df['City Name'].value_counts().head(6).index
    city_avg = df[df['City Name'].isin(top_cities)].groupby('City Name')['Average Price'].mean().sort_values()
    sns.barplot(x=city_avg.index, y=city_avg.values, palette='viridis')
    plt.title('主要城市的南瓜平均价格')
    plt.xlabel('城市')
    plt.ylabel('平均价格')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'city_price.png'))
    plt.close()

def plot_monthly_trend(df):
    """绘制月度价格趋势图"""
    monthly_avg = df.groupby('Month')['Average Price'].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_avg.index, monthly_avg.values, marker='o', color='green', linewidth=2)
    plt.title('南瓜价格月度趋势')
    plt.xlabel('月份')
    plt.ylabel('平均价格')
    plt.xticks(range(1, 13))
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(IMAGES_DIR, 'monthly_trend.png'))
    plt.close()

def plot_size_price(df):
    """绘制大小与价格的关系散点图"""
    plt.figure(figsize=(10, 6))
    size_mapping = {'sml': 1, 'med': 2, 'lge': 3, 'xlge': 4, 'jbo': 5}
    df['Size Code'] = df['Item Size'].map(size_mapping).dropna()
    sns.scatterplot(x='Size Code', y='Average Price', hue='Variety', data=df, alpha=0.6)
    plt.title('南瓜大小与价格的关系')
    plt.xlabel('大小（1=小，5=特大）')
    plt.ylabel('平均价格')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'size_price.png'))
    plt.close()

def plot_predicted_vs_actual(y_true, y_pred, model_name):
    """绘制预测值与实际值对比图"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title(f'{model_name} 预测值 vs 实际值')
    plt.xlabel('实际价格')
    plt.ylabel('预测价格')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(IMAGES_DIR, f'{model_name}_pred_vs_actual.png'))
    plt.close()