'''
该脚本是调用GPT-4 API接口，分别对生成的知识点、主旨内容、知识点详述以及动态生成的题目进行质量评估。
python 20QualityControl.py \
    --prompt ./prompt/check/KN_check.txt \
    --dataset ./output/KN.json \
    --output ./output/check/KN_check.json
python 20QualityControl.py \
    --prompt ./prompt/check/KN_check.txt \
    --dataset ./output/KN-t.json \
    --output ./output/check/KN_check.json
'''
import argparse
import os  # 新增：用于操作路径
import json  # 新增：用于保存结果为JSON文件
from tqdm import tqdm  # 导入 tqdm 库

from llm_api import call_gpt4_api


def read_txt(txt_path):
    with open(txt_path, "r") as f:
        return f.read()

def read_json(json_path):
    if os.path.exists(json_path):  # 检查文件是否存在
        with open(json_path, "r", encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"文件 {json_path} 不存在.")
        return {}

    
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='./prompt/check/KN_check.txt', type=str)
    parser.add_argument('--dataset', default='./output/KN.json', type=str)
    parser.add_argument('--output', default='./output/check/KN_check.json', type=str)
    return parser.parse_args()

def process():
    print(f"开始处理文件: {configs.dataset}")
    
    data = read_json(configs.dataset)
    
    result_list = []
    
    for index, row in enumerate(tqdm(data, desc="Processing LLM Q&A", ncols=100, unit="item")):
        prompt = read_txt(configs.prompt).replace("{{KN}}", str(row))
        
        # 重试逻辑，最多3次
        retry_count = 0
        while retry_count < 3:
            res = eval(call_gpt4_api(prompt))  # 调用大模型 API
            if isinstance(res, list):
                result_list.append(res[0])
                break
            else:
                print(f"警告: 输出无效 ({res})，正在重新请求...")
                retry_count += 1
            if retry_count == 3:
                print(f"警告: 超过3次重试，跳过当前项: {res}")
            
        # 将 数据写入 JSON 文件
        with open(configs.output, "w", encoding="utf-8") as f:
            json.dump(result_list, f, ensure_ascii=False, indent=4)
            
if __name__ == "__main__":
    configs = argparser()  # 加载参数配置
    process()