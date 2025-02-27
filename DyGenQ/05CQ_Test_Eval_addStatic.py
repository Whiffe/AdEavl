'''
通过在提示词中加入静态数据模拟污染
'''
from llm_api import call_gpt3_api, call_gpt4_api, call_doub_api, call_qwen_api, call_claude3_api, call_glm4_api, call_llama3_api, call_deepseek_api
import argparse
import json
from tqdm import tqdm  # 导入 tqdm 库
import os
import math

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='./promptEn/CQTestAddStatic.txt', type=str)
    parser.add_argument('--dataset', default='./output/CQ_format.json', type=str)
    parser.add_argument('--datasetStatic', default='./dataset/formatDataset/mmlu/mmlu_sample_300.json', type=str)
    parser.add_argument('--model', default='gpt3', type=str)
    parser.add_argument('--output', default='./output/CQEdEaval_AddStatic_gpt3_Answer.json', type=str)
    parser.add_argument('--reconstruct', action='store_true') 
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

def process():
    print(f"开始处理文件: {configs.dataset}")


    data = read_json(configs.dataset)
    dataStatic = read_json(configs.datasetStatic)
    len_data = len(data)
    len_dataStatic = len(dataStatic)

    # 用于存储所有的 SAQ 数据
    cq_list = []
    
    # 使用 tqdm 创建进度条
    with tqdm(total=len(data), desc="Processing questions") as pbar:
        # 遍历数据并将每一行合成为一个字典
        for index, row in enumerate(data):
            # 重试逻辑，最多3次
            retry_count = 0
            while retry_count < 3:
                index_dataStatic = math.floor(
                    index*(len_dataStatic/len_data)
                )
                
                prompt = read_txt(configs.prompt).replace("{{choiceQ}}", row['instruction']).replace("{{staticchoiceQ}}", dataStatic[index_dataStatic]['instruction']).replace("{{staticAnswer}}", dataStatic[index_dataStatic]['output'])
                # print(prompt)
                # input()
                # 调用大模型 API
                if configs.model == 'gpt3':
                    res = call_gpt3_api(prompt)
                if configs.model == 'gpt4':
                    res = call_gpt4_api(prompt)
                if configs.model == 'doub':
                    res = call_doub_api(prompt)
                if configs.model == 'qwen':
                    res = call_qwen_api(prompt)
                if configs.model == 'claude3':
                    res = call_claude3_api(prompt)
                if configs.model == 'glm4':
                    res = call_glm4_api(prompt)
                if configs.model == 'llama3':
                    res = call_llama3_api(prompt)
                if configs.model == 'deepseek':
                    res = call_deepseek_api(prompt)
                    # print(res)
                    
                
                # 判断返回的分类是否有效（A/B/C/D）
                if res in ["A", "B", "C", "D"]:
                    if res == row['output']:
                        cq_answer = '1'
                    else:
                        cq_answer = '0'
                    if configs.reconstruct and 'choiceQ' in row:
                        cq_item = {
                            "instruction": row['instruction'],
                            "output": row['output'],
                            "answer": res,
                            "score": cq_answer,
                            "choiceQ": row["choiceQ"],
                            "KN": row["KN"],
                            "purport": row["purport"],
                            "KNexplain": row["KNexplain"]
                        }
                    else:
                        cq_item = {
                            "instruction": row['instruction'],
                            "output": row['output'],
                            "answer": res,
                            "score": cq_answer
                        }
                    if 'Bloom' in configs.dataset:
                        cq_item = {
                            "Layer": row['Layer'],
                            "instruction": row['instruction'],
                            "output": row['output'],
                            "answer": res,
                            "score": cq_answer,
                            "choiceQ": row["choiceQ"],
                            "KN": row["KN"],
                            "purport": row["purport"],
                            "KNexplain": row["KNexplain"]
                        }
                    cq_list.append(cq_item)
                    break  # 如果分类有效，则跳出循环
                else:
                    # 如果分类无效，打印提示并重试
                    print(f"警告: 分类结果无效 ({res})，正在重新请求...")
                    retry_count += 1
                if retry_count == 3:
                    # 如果尝试3次都无效，打印警告并跳过当前项
                    print(f"警告: 超过3次重试，跳过当前 Q&A: {res}")
                    break
        
            # 更新进度条
            pbar.update(1)
            
            # 将 SAQ 数据写入 JSON 文件
            with open(configs.output, "w", encoding="utf-8") as f:
                json.dump(cq_list, f, ensure_ascii=False, indent=4)
                

if __name__ == "__main__":
    configs = argparser()  # 加载参数配置
    process()
