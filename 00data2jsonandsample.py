'''
python 00data2jsonandsample.py \
    --dataset dataset/mmlu/data/test \
    --output_path dataset/formatDataset/mmlu/mmlu_sample_300.json \
    --sample_size 300

python 00data2jsonandsample.py \
    --dataset dataset/ai2_arc/ARC-Challenge/test-00000-of-00001.parquet \
    --output_path dataset/formatDataset/ai2_arc/arc_sample_300.json \
    --sample_size 300
'''

import pyarrow.parquet as pq
import pandas as pd
import json
from tqdm import tqdm
import numpy as np
import argparse
import os

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset/mmlu/data/test', type=str)
    parser.add_argument('--output_path', default='dataset/formatDataset/mmlu/mmlu_sample_300.json', type=str)
    parser.add_argument('--sample_size', default=300, type=int)
    return parser.parse_args()

def mmlu():
    input_path = configs.dataset
    
    output_path = configs.output_path
        
    # 获取目录中的所有文件
    all_files = os.listdir(input_path)
    
    # 筛选出 CSV 文件
    csv_files = [f for f in all_files if f.endswith('.csv')]
    
    # 如果没有 CSV 文件
    if not csv_files:
        print(f"目录 {input_path} 中没有 CSV 文件！")
        return []
    
    saq_list = []
    print(f"开始读取目录 {input_path} 下的 {len(csv_files)} 个 CSV 文件...")
    count_nums = 0
    # 遍历并读取 CSV 文件
    for csv_file in csv_files:
        csv_path = os.path.join(input_path, csv_file)
        try:
            df = pd.read_csv(csv_path, header=None)
            print(f"读取{csv_path}")
            for index, row in df.iterrows():
                question = f"{row[0]} \nA {row[1]} \nB {row[2]} \nC {row[3]} \nD {row[4]}"
                output = row[5]
                
                saq_entry = {
                    "instruction": question,
                    "input": "",
                    "output": output,
                    "system": "",
                    "history": []
                }
                saq_list.append(saq_entry)
                count_nums += 1
                
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(saq_list, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"读取文件 {csv_file} 时出错: {e}")
    print(f"处理了：{count_nums} 条数据")
    
    return saq_list
def ai2_arc():
    input_path = configs.dataset
    output_path = configs.output_path
    
    if not input_path.endswith('.parquet'):
        print(f"指定的文件 {input_path} 不是一个 Parquet 文件！")
        return []
    
    saq_list = []
    
    df = pd.read_parquet(input_path)
    print(f"读取 {input_path}")

    for index, row in df.iterrows():
        question = f"{row['question']}"
        
        # 动态拼接选项
        options = ""
        for i in range(len(row['choices']['label'])):
            options += f"{row['choices']['label'][i]} {row['choices']['text'][i]}\n"
        
        # 合并问题和选项
        question = f"{question}\n{options.strip()}"
        
        output = row['answerKey']

        saq_entry = {
            "instruction": question,
            "input": "",
            "output": output,
            "system": "",
            "history": []
        }
        saq_list.append(saq_entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(saq_list, f, ensure_ascii=False, indent=4)

    print(f"处理了：{len(saq_list)} 条数据")

    return saq_list


def sample_from_jsondata(data):
    # 获取总数据量
    total_records = len(data)
    print(f"总记录数: {total_records}")
    
    sample_size = configs.sample_size
    
    # 检查样本量是否小于总记录数
    if sample_size > total_records:
        print(f"样本数量 ({sample_size}) 超过总记录数 ({total_records})，将使用全部数据。")
        sampled_data = data
    else:
        # 使用 numpy 进行均匀采样
        indices = np.linspace(0, total_records - 1, sample_size, dtype=int)
        sampled_data = [data[i] for i in indices]
        
    print(sampled_data)
    print(f"采样了 {len(sampled_data)} 条数据")
    
    output_path = configs.output_path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=4)


def process():
    """
    从 Parquet 文件均匀采样指定数量的数据并转换为 JSON 格式保存。

    :param input_parquet_path: str, 输入的 Parquet 文件路径
    :param output_json_path: str, 输出的 JSON 文件路径
    :param sample_size: int, 需要采样的记录数量
    """
    if 'mmlu' in configs.dataset:
        json_data = mmlu()
    if 'arc' in configs.dataset:
        json_data = ai2_arc()
    sample_from_jsondata(json_data)

if __name__ == "__main__":
    configs = argparser()  # 加载参数配置
    process()