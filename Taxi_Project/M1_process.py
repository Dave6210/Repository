import pandas as pd
import numpy as np


def load_and_clean_data(file_path):
    print("正在执行 M1: 数据处理与质量分析...")
    df = pd.read_parquet(file_path, engine='pyarrow')

    #1.生成数据质量报告
    print("\n" + "=" * 30)
    print("--- 数据质量报告 ---")
    print(f"总记录数: {len(df)}")
    print(f"行程距离异常(<=0): {len(df[df['trip_distance'] <= 0])}")
    print(f"车费异常(<=0): {len(df[df['fare_amount'] <= 0])}")

    print("\n[缺失值统计明细]:")
    null_counts = df.isnull().sum()
    #使用循环实现“一行一个字段”输出
    for column, count in null_counts.items():
        #使用ljust让列名左对齐，看起来更整齐
        print(f"  - {column.ljust(25)} : {count} 个缺失值")

    print("=" * 30 + "\n")

    #2.清洗数据并在注释中说明理由
    #策略1:剔除无效行程。距离或车费为0的数据无法反映真实的交通需求。
    df = df[(df['fare_amount'] > 0) & (df['trip_distance'] > 0)]
    #策略2:时间窗口过滤。确保分析的是2023年1月的数据，排除录入错误的年份。
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df = df[(df['tpep_pickup_datetime'].dt.year == 2023) & (df['tpep_pickup_datetime'].dt.month == 1)]
    #策略3:剔除极端长途。超过100英里的行程在城市出租车中属于极少数离群点。
    df = df[df['trip_distance'] < 100]

    #3.特征提取
    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    #高峰时段定义:早7-9点，晚17-19点
    df['is_peak'] = df['hour'].apply(lambda x: 1 if (7 <= x <= 9 or 17 <= x <= 19) else 0)

    #4.设计2个有意义的衍生特征
    #特征A:平均每英里费率 (反映路段拥堵或加价情况)
    df['fare_per_mile'] = df['fare_amount'] / df['trip_distance']
    #特征B:行程时长(分钟)
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

    return df