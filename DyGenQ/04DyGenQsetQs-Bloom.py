''''
一个问题生成多个，步骤一步步变得复杂
第四步是，根据讲解的知识点，来出选择题和简答题。
'''
import pyarrow.parquet as pq
from llm_api import call_qwen_api, call_qwen_net_api, call_doub_api  # 假设你使用这个 API 函数
import argparse
import json
from tqdm import tqdm  # 导入 tqdm 库
import os

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='./promptEn/setCQ_Bloom.txt', type=str)
    parser.add_argument('--dataset', default='./output/mmlu/Reconstruct_Complex/DyCQ_Reconstructed_mmlu_sample_300.json', type=str)
    parser.add_argument('--model', default='Doub', type=str)
    parser.add_argument('--outputCQ', default='./output/mmlu/DyCQBloom_mmlu_sample_300.json', type=str)
    parser.add_argument('--fewshot', default='./promptEn/few-shot/setCQ_Bloom_mmlu_sample_300.txt', type=str)
    return parser.parse_args()

def safe_json_parse(response_str):
    """
    Try to parse a response string to JSON. If it fails due to unescaped backslashes,
    escape them and retry parsing.
    """
    try:
        return json.loads(response_str)  # First attempt
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        # If JSONDecodeError occurs, try escaping backslashes
        response_str_escaped = response_str.replace("\\", "\\\\")
        try:
            return json.loads(response_str_escaped)  # Retry parsing with escaped backslashes
        except json.JSONDecodeError as e2:
            print(f"JSONDecodeError after escaping: {e2}")
            return None

def read_txt(txt_path):
    with open(txt_path, "r") as f:
        return f.read()

def clear_file(file_path):
    """清空文件内容，如果文件不存在则创建"""
    with open(file_path, "w", encoding="utf-8") as f:
        pass

def read_json(json_path):
    if os.path.exists(json_path):  # 检查文件是否存在
        with open(json_path, "r", encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"文件 {json_path} 不存在.")
        return {}

def process(file_path, outputCQ):
    print(f"开始处理文件: {file_path}")    # 清空输出文件内容
    clear_file(outputCQ)
    data = read_json(file_path)

    # 用于存储生成的CQ_format问题
    CQ_format = []
    # 使用 tqdm 创建进度条
    for index, row in enumerate(tqdm(data, desc="Processing LLM Q&A", ncols=100, unit="item")):

        KNexplain = row['KNexplain']
        purport = row['purport']
        KN = row['KN']
        choiceQ = row['instruction'] + '\n Answer: ' + row['output']

        prompt = read_txt(configs.prompt).replace("{{KNexplain}}", KNexplain).replace("{{purport}}", purport).replace("{{choiceQ}}", choiceQ).replace("{{few-shot}}", read_txt(configs.fewshot)).replace("{{kn}}", KN)

        # 重试逻辑，最多重试 3 次
        retry_count = 0
        while retry_count < 3:
            try:
                if configs.model == 'Qwen':
                    responseCQ  = call_qwen_api(prompt)  # 调用大模型 API
                if configs.model == 'Doub':
                    responseCQ  = call_doub_api(prompt)
                # 将responseCQ与responseSAQ从str转化为json
                # 这一步不一定能转化成功，如果转化失败，则retry_count += 1,重新调用大模型回答
                CQ_json = safe_json_parse(responseCQ)
                
                if CQ_json is None:
                    raise ValueError("Response is not in valid JSON format.")
                    retry_count += 1
                    print(f"警告: 非json格式 ({CQ_json},{SAQ_json})，正在重试 ({retry_count}/3)")
                else:
                    for i, cq in enumerate(CQ_json):
                        CQ_json_format = {
                            "Layer": cq['Layer'],
                            "instruction": f"{cq['Question']} \nA {cq['A']} \nB {cq['B']} \nC {cq['C']} \nD {cq['D']} ",
                            "input": "",
                            "output": cq['Answer'],
                            "system": "",
                            "history": [],
                            "choiceQ": choiceQ,
                            "KN": KN,
                            "purport": purport,
                            "KNexplain": KNexplain
                        }
                        
                        # 成功解析后，将结果存储
                        CQ_format.append(CQ_json_format)

                        
                    break  # 成功后退出重试循环
                    

            except Exception as e:
                retry_count += 1
                print(f"警告: 请求失败 ({e})，正在重试 ({retry_count}/3)")

                print("prompt:",prompt)
                print("responseCQ:",responseCQ)
                print("CQ_json:",CQ_json)
                print("------")

        if retry_count == 3:
            print("Error: Unable to generate explanation after 3 attempts")
            # KNexplain.append("Error: Unable to generate explanation after 3 attempts")
            # print("promptCQ:",promptCQ)
            print("responseCQ:",responseCQ)
            print("------")
        with open(outputCQ, "w", encoding="utf-8") as f:
            json.dump(CQ_format, f, ensure_ascii=False, indent=4)




if __name__ == "__main__":
    configs = argparser()  # 加载参数配置
    process(configs.dataset, configs.outputCQ)
