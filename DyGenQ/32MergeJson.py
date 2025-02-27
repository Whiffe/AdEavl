'''
将重构的数据与原来剩下没被选中的数据进行合并
python 32MergeJson.py \
    --json1 ./output/mmlu/Reconstruct_Complex/Reconstructed_college_computer_science_test.json \
    --json2 ./output/mmlu/Reconstruct_Complex/Remain_college_computer_science_test.json \
    --output ./output/mmlu/Reconstruct_Complex/CQ_Reconstructed_college_computer_science_test.json
'''

import json
import sys
import argparse
def argparser():
    # 定义并解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--json1', default='.', type=str)
    parser.add_argument('--json2', default='.', type=str)
    parser.add_argument('--output', default='', type=str)

    return parser.parse_args()
    
def merge_json_files(file1, file2):
    # 读取第一个JSON文件
    with open(file1, 'r') as f1:
        data1 = json.load(f1)
    
    # 读取第二个JSON文件
    with open(file2, 'r') as f2:
        data2 = json.load(f2)
    
    # 合并两个数组
    merged_data = data1 + data2
    
    # 将合并后的数据写入新文件
    with open(configs.output, 'w') as outfile:
        json.dump(merged_data, outfile, indent=4)

if __name__ == "__main__":
    # 解析命令行参数
    configs = argparser()
    
    merge_json_files(configs.json1, configs.json2)
    print(f"JSON files have been merged into {configs.output}")
