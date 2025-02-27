'''
python 08countScoreSAQ.py \
    --json ./output/org/SAQ_Answer.json
python 08countScoreSAQ.py \
    --json ./output/org/SAQ_Answer_SAQ_format.json
    
python 08countScoreSAQ.py \
    --json ./output/IFT/SAQ_Answer.json
python 08countScoreSAQ.py \
    --json ./output/IFT/SAQ_Answer_SAQ_format.json
'''

import json
import argparse
import os

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', default='./output/org/CQ_Answer_cmmlu0.0125.json', type=str)
    return parser.parse_args()

def read_json(json_path):
    if os.path.exists(json_path):  # 检查文件是否存在
        with open(json_path, "r", encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"文件 {json_path} 不存在.")
        return {}


def process(file_path):
    data = read_json(file_path)
    score = 0
    for index, row in enumerate(data):
        score += int(row['score'])
    print(f'{file_path} score : {score}')
    
if __name__ == "__main__":
    configs = argparser()  # 加载参数配置
    process(configs.json)