import json
import argparse
import os
from collections import defaultdict

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
    print(f'{score}/{len(data)}={score/len(data):.3f}')

    if 'Bloom' in file_path:
        layer_scores = defaultdict(lambda: {'score': 0, 'count': 0})
        for row in data:
            layer = row.get('Layer', 'Unknown')
            score = int(row.get('score', 0))
            layer_scores[layer]['score'] += score
            layer_scores[layer]['count'] += 1
        total_score = 0
        total_count = 0
        fixed_order = ['Remembering', 'Understanding', 'Applying', 'Analyzing', 'Evaluating', 'Creating']
        for layer in fixed_order:
            stats = layer_scores.get(layer, {'score': 0, 'count': 0})
            total_score += stats['score']
            total_count += stats['count']
            print(f"{layer} {stats['score']}/{stats['count']}={stats['score'] / stats['count'] if stats['count'] > 0 else 0:.3f}")
    
        print(f"\nOverall {total_score}/{total_count}={total_score / total_count if total_count > 0 else 0:.3f}")

if __name__ == "__main__":
    configs = argparser()  # 加载参数配置
    process(configs.json)
