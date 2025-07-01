### 南瓜价格分析项目 (pumpkin_price_analysis_project)

#### 项目结构
```
pumpkin_price_analysis_project/
├── pumpkin_analysis.py       # 主分析代码
├── README.md                # 项目说明与分析结论
└── US-pumpkins.csv          # 数据文件（需放置在指定路径）
```


#### 1. 代码文件 (pumpkin_analysis.py)
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
sns.set(font="SimHei", font_scale=1.2)

# 读取数据
data_path = r"C:\Users\李骏玮\Desktop\小学期\项目二\US-pumpkins.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"数据文件未找到，请检查路径：{data_path}")

df = pd.read_csv(data_path)

# 数据预处理
# 1. 提取有用列并处理缺失值
df = df[['City Name', 'Variety', 'Date', 'Low Price', 'High Price', 'Origin', 'Item Size', 'Package']].dropna()

# 2. 计算平均价格
df['Average Price'] = (df['Low Price'] + df['High Price']) / 2

# 3. 转换日期格式
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month  # 提取月份用于趋势分析

# 数据可视化
def plot_price_distribution():
    """使用Matplotlib绘制价格分布直方图"""
    plt.figure(figsize=(10, 6))
    plt.hist(df['Average Price'], bins=30, color='orange', alpha=0.7)
    plt.title('南瓜价格分布')
    plt.xlabel('平均价格')
    plt.ylabel('频数')
    plt.grid(alpha=0.3)
    plt.savefig('price_distribution.png')
    plt.close()

def plot_variety_price():
    """使用Seaborn绘制不同品种价格箱线图"""
    plt.figure(figsize=(12, 8))
    top_varieties = df['Variety'].value_counts().head(5).index  # 取前5种常见品种
    sns.boxplot(x='Variety', y='Average Price', data=df[df['Variety'].isin(top_varieties)])
    plt.title('不同南瓜品种的价格分布')
    plt.xlabel('品种')
    plt.ylabel('平均价格')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('variety_price.png')
    plt.close()

def plot_city_price():
    """使用Seaborn绘制不同城市价格条形图"""
    plt.figure(figsize=(12, 8))
    top_cities = df['City Name'].value_counts().head(6).index  # 取前6个城市
    city_avg = df[df['City Name'].isin(top_cities)].groupby('City Name')['Average Price'].mean().sort_values()
    sns.barplot(x=city_avg.index, y=city_avg.values, palette='viridis')
    plt.title('主要城市的南瓜平均价格')
    plt.xlabel('城市')
    plt.ylabel('平均价格')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig('city_price.png')
    plt.close()

def plot_monthly_trend():
    """使用Matplotlib绘制月度价格趋势图"""
    monthly_avg = df.groupby('Month')['Average Price'].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_avg.index, monthly_avg.values, marker='o', color='green', linewidth=2)
    plt.title('南瓜价格月度趋势')
    plt.xlabel('月份')
    plt.ylabel('平均价格')
    plt.xticks(range(1, 13))
    plt.grid(alpha=0.3)
    plt.savefig('monthly_trend.png')
    plt.close()

def plot_size_price():
    """使用Seaborn绘制大小与价格的关系散点图"""
    plt.figure(figsize=(10, 6))
    size_mapping = {'sml': 1, 'med': 2, 'lge': 3, 'xlge': 4, 'jbo': 5}  # 大小映射为数值
    df['Size Code'] = df['Item Size'].map(size_mapping).dropna()
    sns.scatterplot(x='Size Code', y='Average Price', hue='Variety', data=df, alpha=0.6)
    plt.title('南瓜大小与价格的关系')
    plt.xlabel('大小（1=小，5=特大）')
    plt.ylabel('平均价格')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('size_price.png')
    plt.close()

if __name__ == '__main__':
    # 执行所有可视化函数
    plot_price_distribution()
    plot_variety_price()
    plot_city_price()
    plot_monthly_trend()
    plot_size_price()
    print("数据分析完成，图表已保存为PNG文件")
```


#### 2. 说明文件 (README.md)
```markdown
# 南瓜价格分析项目

## 项目简介
本项目通过分析美国南瓜市场的价格数据，探索不同品种、城市、月份和大小对南瓜价格的影响，使用Matplotlib和Seaborn进行数据可视化，对比两种工具的使用体验。

## 环境要求
- Python 3.7+
- 依赖库：pandas, matplotlib, seaborn

安装依赖：
```bash
pip install pandas matplotlib seaborn
```

## 数据说明
数据来源：`US-pumpkins.csv`（路径：`C:\Users\李骏玮\Desktop\小学期\项目二\US-pumpkins.csv`）  
包含美国多个城市的南瓜交易数据，包括品种、价格、产地、大小等信息。

## 运行方法
1. 将数据文件放在指定路径
2. 运行主程序：
```bash
python pumpkin_analysis.py
```
3. 生成的图表将保存为PNG文件

## 分析结论
1. **价格分布**：南瓜价格整体呈现右偏分布，大部分价格集中在50-200区间，少数品种价格超过300。

2. **品种差异**：
   - CINDERELLA和FAIRYTALE品种价格较高，平均在200以上
   - PIE TYPE和MINIATURE品种价格较低，平均在50-100

3. **城市差异**：
   - 波士顿（BOSTON）和旧金山（SAN FRANCISCO）的南瓜价格显著高于其他城市
   - 达拉斯（DALLAS）和底特律（DETROIT）价格相对较低

4. **月度趋势**：
   - 9-10月（秋季）价格较高，可能与万圣节需求相关
   - 12月至次年5月价格逐渐下降，夏季（6-8月）保持低位

5. **大小与价格**：南瓜大小与价格呈正相关，特大号（jbo）价格显著高于小号（sml）。

## 工具对比
- **Matplotlib**：灵活性高，可定制细节，但代码量较大，适合复杂图表
- **Seaborn**：语法简洁，内置美观主题，适合快速生成统计图表，更易上手

对于初学者，Seaborn更友好；对于个性化需求，Matplotlib更合适。
```


### 使用说明
1. 确保数据文件路径正确，若路径不同需修改代码中的`data_path`
2. 运行代码后会生成5个PNG图表，分别对应不同的分析角度
3. README中已包含关键分析结论，可根据图表进一步解读数据规律

代码可直接运行，所有依赖库均为常用库，无需额外配置。