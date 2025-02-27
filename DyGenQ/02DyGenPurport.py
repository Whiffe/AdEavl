''''
一个问题生成多个，步骤一步步变得复杂
第一步是，把题目中的主旨总结出来

'''
from llm_api import call_qwen_api  # 假设你使用这个 API 函数
import argparse
import json
from tqdm import tqdm  # 导入 tqdm 库
import os

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='./promptEn/choicePurport.txt', type=str)
    parser.add_argument('--dataset', default='./output/KN.json', type=str)
    parser.add_argument('--output', default='./output/Purport.json', type=str)
    parser.add_argument('--fewshot', default='./output/Purport.json', type=str)
    return parser.parse_args()

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

def process(file_path, output_path):
    print(f"开始处理文件: {file_path}")
    data = read_json(file_path)

    # 用于存储所有的 Purport 数据
    purport_list = []

    # 使用 tqdm 创建进度条
    for index, row in enumerate(tqdm(data, desc="Processing LLM Q&A", ncols=100, unit="item")):
        choiceQ = row["choiceQ"]

        prompt = read_txt(configs.prompt).replace("{{choiceQ}}", choiceQ).replace("{{few-shot}}", read_txt(configs.fewshot))
        
        res = call_qwen_api(prompt)  # 调用大模型 API

        # 将结果存入列表
        purport_entry = {
            "choiceQ": row["choiceQ"],
            "KN": row["KN"],
            "purport": res
        }
        purport_list.append(purport_entry)

        # 将数据写入 JSON 文件
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(purport_list, f, ensure_ascii=False, indent=4)
    print(f"数据已成功保存到 {output_path}")

if __name__ == "__main__":
    configs = argparser()  # 加载参数配置
    process(configs.dataset, configs.output)
