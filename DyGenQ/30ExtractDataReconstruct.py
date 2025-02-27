'''
指定的难度偏移量 (delta) 从 JSON 数据集中筛选出特定分数的题目，随机抽取一定数量的题目保存为一个新文件，同时将未被抽取的题目保存到另一个文件中，确保两者之和等于原始数据集。

python 30ExtractDataReconstruct.py \
    --delta -0.15 \
    --dataset ./output/mmlu/CQ_Answer_CQ_format-college_computer_science_test.json \
    --output_select ./output/mmlu/Reconstruct_Complex/Reconstruct_college_computer_science_test.json \
    --output_remain ./output/mmlu/Reconstruct_Complex/Remain_college_computer_science_test.json

'''
import argparse
import json
from tqdm import tqdm
import os
import random

def argparser():
    # 定义并解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=float, help='设置题目难度偏移量')
    parser.add_argument('--dataset', default='./output/mmlu/CQ_Answer_CQ_format-college_computer_science_test.json', type=str, help='输入JSON数据集路径')
    parser.add_argument('--output_select', default='./output/mmlu/Reconstruct_Complex/Reconstruct_college_computer_science_test.json', type=str)
    parser.add_argument('--output_remain', default='./output/mmlu/Reconstruct_Complex/Remain_college_computer_science_test.json', type=str)
    return parser.parse_args()

def read_json(json_path):
    # 读取JSON文件
    if os.path.exists(json_path):
        with open(json_path, "r", encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"文件 {json_path} 不存在。")
        return []

def process():
    # 读取数据集
    data = read_json(configs.dataset)
    if not data:
        return
    
    # 根据 delta 值确定选择的分数
    if configs.delta > 0:
        # delta > 0 表示题目太难，选择答错的题目
        chooseScore = '0'
    else:
        # delta <= 0 表示题目太简单，选择答对的题目
        chooseScore = '1'
    
    # 计算需要选择的题目数量
    chooseNum = int(abs(configs.delta) * len(data))
    print(f"选择 {chooseNum} 道分数为 {chooseScore} 的题目")
    
    # 筛选符合分数的题目
    filtered_data = [item for item in data if item.get('score') == chooseScore]
    
    # 如果匹配的题目不足，选择所有符合条件的题目
    if len(filtered_data) < chooseNum:
        print(f"匹配到的题目数量不足，仅找到 {len(filtered_data)} 道。")
        selected_data = filtered_data
    else:
        # 随机选择指定数量的题目
        selected_data = random.sample(filtered_data, chooseNum)
    
    print(f"最终选择了 {len(selected_data)} 道题目。")
    
    # 计算剩余数据
    remaining_data = [item for item in data if item not in selected_data]
    
    # 将筛选后的题目保存到新文件中
    output_path_selected = configs.output_select
    with open(output_path_selected, 'w', encoding='utf-8') as f:
        json.dump(selected_data, f, indent=4, ensure_ascii=False)
    print(f"筛选后的题目已保存到 {output_path_selected}")
    
    # 将剩余的题目保存到另一个文件中
    output_path_remaining = configs.output_remain
    with open(output_path_remaining, 'w', encoding='utf-8') as f:
        json.dump(remaining_data, f, indent=4, ensure_ascii=False)
    print(f"剩余的题目已保存到 {output_path_remaining}")

if __name__ == "__main__":
    # 解析命令行参数
    configs = argparser()
    # 处理数据集
    process()

