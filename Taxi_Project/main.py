# main.py
from M1_process import load_and_clean_data
from M2_visualize import run_visualizations
from M3_model import train_and_compare
from M4_qa import start_qa
import pandas as pd

if __name__ == "__main__":
    data_path = "data/yellow_tripdata_2023-01.parquet"

    print("=== 第一阶段：数据加载与清洗 (M1) ===")
    df = load_and_clean_data(data_path)

    print("\n=== 第二阶段：分析可视化 (M2) ===")
    run_visualizations(df)

    print("\n=== 第三阶段：模型训练 (M3) ===")
    # 抽样 10 万条防止内存溢出
    df_sample = df.sample(n=100000, random_state=42)
    model, rf = train_and_compare(df_sample)

    print("\n=== 第四阶段：智能问答系统 (M4) ===")
    start_qa(df, model)