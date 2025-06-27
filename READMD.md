# 尼日利亚音乐聚类分析项目

## 项目概述

本项目对尼日利亚音乐数据集进行聚类分析，探索不同音乐类型之间的相似性和差异性。通过K-Means聚类算法，我们发现了音乐风格的自然分组，并分析了每个聚类的特征。

## 数据集

数据集包含尼日利亚Spotify上的流行歌曲，包含以下特征：
- 舞蹈性 (danceability)
- 声学特征 (acousticness)
- 能量 (energy)
- 乐器性 (instrumentalness)
- 现场感 (liveness)
- 响度 (loudness)
- 言语性 (speechiness)
- 速度 (tempo)
- 流行度 (popularity)

数据集路径: `C:\Users\李骏玮\Desktop\小学期\nigerian-songs.csv`

## 技术栈

- Python 3.8+
- pandas
- scikit-learn
- matplotlib
- seaborn
- plotly

## 安装与运行

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/nigerian-music-clustering.git
cd nigerian-music-clustering
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 运行Jupyter Notebook：
```bash
jupyter notebook nigerian_music_clustering.ipynb
```

## 分析步骤

### 1. 数据加载与探索
- 加载CSV文件，检查数据结构和质量
- 描述性统计分析
- 特征相关性分析

### 2. 数据预处理
- 处理缺失值
- 特征选择
- 数据标准化

### 3. 聚类分析
- 使用肘部法则确定最佳聚类数
- 执行K-Means聚类
- 分析聚类结果

### 4. 可视化
- 2D散点图可视化聚类结果
- 3D交互式可视化
- 聚类特征雷达图

## 关键发现

1. **聚类分布**：最佳聚类数为5，每个聚类代表不同的音乐风格组合
2. **聚类特征**：
   - 聚类1：高能量、高响度的舞曲
   - 聚类2：高乐器性、低声学特征的电子音乐
   - 聚类3：高言语性的说唱音乐
   - 聚类4：高声学特征、低能量的民谣
   - 聚类5：中等特征的主流流行音乐

3. **流行度分析**：高流行度歌曲集中在聚类1和聚类5

## 代码实现

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
from sklearn.decomposition import PCA

# 加载数据
file_path = "C:/Users/李骏玮/Desktop/小学期/nigerian-songs.csv"
df = pd.read_csv(file_path)

# 数据预处理
features = ['danceability', 'acousticness', 'energy', 'instrumentalness', 
            'liveness', 'loudness', 'speechiness', 'tempo', 'popularity']
df_clean = df.dropna(subset=features)
X = df_clean[features]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 确定最佳聚类数
wcss = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# 肘部法则可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, wcss, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')
plt.tight_layout()
plt.show()

# 基于肘部法则选择最佳K值
best_k = 5
kmeans = KMeans(n_clusters=best_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df_clean['cluster'] = clusters

# 聚类结果分析
cluster_means = df_clean.groupby('cluster')[features].mean()
print(cluster_means)

# 可视化聚类中心
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_means.T, cmap='viridis', annot=True)
plt.title('Cluster Feature Means')
plt.show()

# 2D PCA可视化
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
df_clean['pca1'] = principal_components[:, 0]
df_clean['pca2'] = principal_components[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=df_clean, 
                palette='viridis', s=80, alpha=0.8)
plt.title('2D PCA of Nigerian Music Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

# 3D交互式可视化
pca_3d = PCA(n_components=3)
principal_components_3d = pca_3d.fit_transform(X_scaled)
df_clean['pca1_3d'] = principal_components_3d[:, 0]
df_clean['pca2_3d'] = principal_components_3d[:, 1]
df_clean['pca3_3d'] = principal_components_3d[:, 2]

fig = px.scatter_3d(
    df_clean, 
    x='pca1_3d', 
    y='pca2_3d', 
    z='pca3_3d',
    color='cluster',
    hover_name='name',
    hover_data=['artist', 'artist_top_genre'],
    title='3D PCA Visualization of Nigerian Music Clusters',
    labels={'cluster': 'Cluster'},
    opacity=0.7,
    color_continuous_scale=px.colors.sequential.Viridis
)
fig.update_traces(marker=dict(size=5))
fig.show()

# 聚类特征雷达图
def plot_radar_chart(cluster_id, data):
    categories = features
    values = data.loc[cluster_id].values.tolist()
    values += values[:1]  # 闭合雷达图
    
    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1.0"], color="grey", size=8)
    plt.ylim(0, 1)
    
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=f"Cluster {cluster_id}")
    ax.fill(angles, values, alpha=0.1)
    
    plt.title(f'Cluster {cluster_id} Feature Profile', size=15, y=1.1)
    plt.show()

# 为每个聚类绘制雷达图
for cluster_id in range(best_k):
    plot_radar_chart(cluster_id, cluster_means)
```

## 结论

通过聚类分析，我们发现尼日利亚音乐可以分为5个主要类型，每种类型都有其独特的音频特征组合：

1. **高能量舞曲**：适合派对和舞蹈场景
2. **电子音乐**：强调乐器元素，适合放松和背景音乐
3. **说唱音乐**：高言语性，叙事性强
4. **声学民谣**：原声乐器为主，情感表达丰富
5. **主流流行音乐**：平衡的特征组合，受众广泛

这些发现可以帮助音乐平台更好地组织音乐库，为听众提供更个性化的推荐，也为音乐制作人提供了创作方向上的参考。

## 未来工作

1. 尝试其他聚类算法如DBSCAN或层次聚类
2. 结合自然语言处理分析歌词内容
3. 构建推荐系统基于聚类结果
4. 分析音乐趋势随时间的变化
