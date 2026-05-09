import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def run_visualizations(df):
    print("正在执行 M2: 分析可视化...")
    if not os.path.exists('outputs'): os.makedirs('outputs')

    #1.出行需求时间规律 (分小时 & 工作日/周末)
    plt.figure(figsize=(10, 5))
    sns.pointplot(data=df, x='hour', y='VendorID', hue='is_weekend', estimator=len, markers=".")
    plt.title('工作日 vs 周末各时段出行需求')
    plt.savefig('outputs/time_pattern.png')

    #2.区域热度分析 (Top 10 上车地点)
    plt.figure(figsize=(10, 5))
    top_zones = df['PULocationID'].value_counts().head(10)
    top_zones.plot(kind='bar', color='skyblue')
    plt.title('Top 10 热门上车区域 ID')
    plt.savefig('outputs/hot_zones.png')

    #3.车费影响因素分析 (距离与费用的关系)
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=df.sample(2000), x='trip_distance', y='fare_amount', alpha=0.5)
    plt.title('行程距离与车费的相关性分析')
    plt.savefig('outputs/fare_factors.png')

    #4.自选分析：不同时段的小费比例波动
    df['tip_ratio'] = (df['tip_amount'] / df['fare_amount']) * 100
    plt.figure(figsize=(10, 5))
    df.groupby('hour')['tip_ratio'].mean().plot(kind='line', marker='s', color='orange')
    plt.title('24小时平均小费比例变化 (%)')
    plt.savefig('outputs/tip_analysis.png')

    print("所有分析图表已保存至 outputs/ 目录。")