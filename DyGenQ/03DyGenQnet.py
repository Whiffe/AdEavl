''''
这一步是，联网，把题目与其对应的知识点，主旨，在网络是搜索整理出对应的知识点讲解。

'''
import pyarrow.parquet as pq
from llm_api import call_qwen_api, call_qwen_net_api  # 假设你使用这个 API 函数
import argparse
import json
from tqdm import tqdm  # 导入 tqdm 库
import os
import random

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='./promptEn/KNexplain.txt', type=str)
    parser.add_argument('--dataset', default='./output/Purport.json', type=str)
    parser.add_argument('--output', default='./output/KNexplain.json', type=str)
    parser.add_argument('--knowledge_points_num', default=None, type=int, help="从知识点中选择的最大数量")
    parser.add_argument('--fewshot', default='./promptEn/few-shot/KNexplain_college_computer_science_dev.txt', type=str)
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

def process(file_path, output_path, knowledge_points_num):
    print(f"开始处理文件: {file_path}")
    try:
        data = read_json(file_path)

        # 存储最终结果
        result_data = []
        
        # 使用 tqdm 创建进度条
        for index, row in enumerate(tqdm(data, desc="Processing LLM Q&A", ncols=100, unit="item")):
            
            choiceQ = row['choiceQ']
            knowledge_points = row['KN']
            purport = row['purport']

            # 根据输入的参数随机选择指定数量的知识点
            if knowledge_points_num is not None:
                if len(knowledge_points) > knowledge_points_num:
                    knowledge_points = random.sample(knowledge_points, knowledge_points_num)

            # 用于存储当前问题的知识点解释
            KNexplain = []
            for kn_i in knowledge_points:
                prompt = read_txt(configs.prompt).replace("{{choiceQ}}", choiceQ).replace("{{kn}}", kn_i).replace("{{purport}}", purport).replace("{{few-shot}}", read_txt(configs.fewshot))
                # 重试逻辑，最多重试 3 次
                retry_count = 0
                while retry_count < 3:
                    try:
                        response  = call_qwen_net_api(prompt)  # 调用大模型 API 联网
                        KNexplain.append(response)  # 将解释加入列表
                        break  # 成功后退出重试循环
                    except Exception as e:
                        retry_count += 1
                        print(f"警告: 请求失败 ({e})，正在重试 ({retry_count}/3)")

                if retry_count == 3:
                    KNexplain.append("Error: Unable to generate explanation after 3 attempts")

            # 将结果添加到最终数据中
            result_data.append({
                "choiceQ": choiceQ,
                "KN": knowledge_points,
                "purport": purport,
                "KNexplain": KNexplain
            })


            # 将 SAQ 数据写入 JSON 文件
            with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=4)
        print(f"SAQ 数据已成功保存到 {output_path}")

    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请检查路径是否正确。")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

if __name__ == "__main__":
    configs = argparser()  # 加载参数配置
    process(configs.dataset, configs.output, configs.knowledge_points_num)
