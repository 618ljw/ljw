import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib as mpl

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号
mpl.rcParams['font.family'] = 'sans-serif'

# 加载数据
file_path = "C:\\Users\\李骏玮\\Desktop\\小学期\\nigerian-songs.csv"
data = pd.read_csv(file_path)

# 1. 数据预处理
print("数据集形状:", data.shape)
print("\n缺失值统计:")
print(data.isnull().sum())

# 选择数值型特征
features = data[['danceability', 'acousticness', 'energy', 'instrumentalness',
                 'liveness', 'loudness', 'speechiness', 'tempo', 'length', 'popularity']]

# 标准化数据
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 2. 确定最佳聚类数
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

    if k > 1:  # 轮廓系数需要至少2个聚类
        score = silhouette_score(scaled_features, kmeans.labels_)
        silhouette_scores.append(score)

# 绘制肘部法则图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'bx-')
plt.xlabel('聚类数 (k)')
plt.ylabel('惯性 (Inertia)')
plt.title('肘部法则')

# 绘制轮廓系数图
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, 'ro-')
plt.xlabel('聚类数 (k)')
plt.ylabel('轮廓系数')
plt.title('轮廓系数分析')
plt.tight_layout()
plt.savefig('聚类分析参数.png', dpi=300, bbox_inches='tight')  # 保存图像避免字体问题
plt.close()  # 关闭图形避免内存泄漏

# 3. 根据轮廓系数选择最佳k值（这里选择k=4）
best_k = 4
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_features)
data['Cluster'] = clusters

# 4. PCA降维可视化
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)
data['PCA1'] = principal_components[:, 0]
data['PCA2'] = principal_components[:, 1]

plt.figure(figsize=(10, 7))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data,
                palette='viridis', s=60, alpha=0.8)
plt.title('音乐聚类可视化 (PCA降维)')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.legend(title='聚类')
plt.savefig('聚类可视化.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. 聚类分析
# 计算每个聚类的特征均值
cluster_means = data.groupby('Cluster')[features.columns].mean()

plt.figure(figsize=(14, 10))
for i, feature in enumerate(features.columns, 1):
    plt.subplot(3, 4, i)
    # 修复FutureWarning
    sns.barplot(x='Cluster', y=feature, data=data, palette='viridis',
                hue='Cluster', legend=False)
    plt.title(feature.capitalize())
plt.tight_layout()
plt.suptitle('聚类特征分析', y=1.02, fontsize=16)
plt.savefig('聚类特征分析.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. 聚类描述
cluster_descriptions = []
for cluster_id in range(best_k):
    cluster_data = data[data['Cluster'] == cluster_id]
    top_artists = cluster_data['artist'].value_counts().head(3).index.tolist()
    top_genres = cluster_data['artist_top_genre'].value_counts().head(3).index.tolist()

    description = {
        'Cluster': cluster_id,
        'Size': len(cluster_data),
        'Avg Popularity': cluster_data['popularity'].mean(),
        'Avg Danceability': cluster_data['danceability'].mean(),
        'Avg Energy': cluster_data['energy'].mean(),
        'Top Artists': ", ".join(top_artists),
        'Top Genres': ", ".join(top_genres)
    }
    cluster_descriptions.append(description)
0
# 打印聚类描述
print("\n聚类分析结果:")
for desc in cluster_descriptions:
    print(f"\n聚类 {desc['Cluster']}:")
    print(f"  包含歌曲数: {desc['Size']}")
    print(f"  平均流行度: {desc['Avg Popularity']:.1f}")
    print(f"  平均舞蹈性: {desc['Avg Danceability']:.2f}")
    print(f"  平均能量: {desc['Avg Energy']:.2f}")
    print(f"  代表艺术家: {desc['Top Artists']}")
    print(f"  代表流派: {desc['Top Genres']}")

# 7. 保存结果
data.to_csv('nigerian-songs-clustered.csv', index=False, encoding='utf-8-sig')  # 添加编码支持
print("\n聚类结果已保存到文件: nigerian-songs-clustered.csv")
