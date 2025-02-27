'''
对数据集进行质量检查
'''
from llm_api import call_gpt3_api, call_gpt4_api, call_doub_api, call_qwen_api, call_claude3_api, call_glm4_api, call_llama3_api, call_qwen_net_api
import argparse
import json
from tqdm import tqdm
import os
from collections import Counter

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='./promptEn/check/Q_check.txt', type=str)
    parser.add_argument('--dataset', default='./output/mmlu/Reconstruct_Complex/DyCQ_Reconstructed_mmlu_sample_300.json', type=str)
    parser.add_argument('--output', default='./output/check/Q_check_DyCQ_Reconstructed_mmlu_sample_300.json', type=str)
    return parser.parse_args()

def read_txt(txt_path):
    with open(txt_path, "r") as f:
        return f.read()

def read_json(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r", encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"文件 {json_path} 不存在.")
        return {}

def call_models_with_retry(prompt, max_retries=3):
    """
    调用三个模型，最多重试 max_retries 次。
    如果结果有效（'0' 或 '1'），返回结果，否则返回 None。
    """
    for retry_count in range(max_retries):
        res_gpt4 = call_gpt4_api(prompt)
        res_doub = call_doub_api(prompt)
        res_qwen = call_qwen_net_api(prompt)
        
        # 检查是否所有结果都有效
        if all(res in ["0", "1"] for res in [res_gpt4, res_doub, res_qwen]):
            return res_gpt4, res_doub, res_qwen
        else:
            print(f"警告: 分类结果无效 (GPT4: {res_gpt4}, Doub: {res_doub}, Qwen: {res_qwen})，正在重新请求...")
    print(f"警告: 超过 {max_retries} 次重试，放弃当前调用.")
    return None, None, None

def determine_final_result(res_gpt4, res_doub, res_qwen):
    """
    根据三个模型的结果决定最终结果。
    """
    results = [res_gpt4, res_doub, res_qwen]
    if results.count("0") >= 2:
        return "0"
    elif results.count("1") >= 2:
        return "1"
    return None

def process():
    print(f"开始处理文件: {configs.dataset}")
    data = read_json(configs.dataset)
    cq_list = []

    for index, row in enumerate(tqdm(data, desc="Processing LLM Q&A", ncols=100, unit="item")):
        prompt = read_txt(configs.prompt).replace("{{choiceQ}}", row['instruction'] + '\nAnswer：' + row['output'])
        res_gpt4, res_doub, res_qwen = call_models_with_retry(prompt)

        if res_gpt4 and res_doub and res_qwen:
            final_res = determine_final_result(res_gpt4, res_doub, res_qwen)
            if final_res:
                cq_item = {
                    "instruction": row['instruction'],
                    "output": row['output'],
                    "Qjudge": final_res
                }
                if 'Bloom' in configs.dataset:
                    cq_item["Layer"] = row['Layer']
                cq_list.append(cq_item)
            else:
                print(f"警告: 无法确定最终结果 (GPT4: {res_gpt4}, Doub: {res_doub}, Qwen: {res_qwen})，跳过.")
        else:
            print(f"警告: 三个模型调用失败，跳过当前 Q&A.")

    with open(configs.output, "w", encoding="utf-8") as f:
        json.dump(cq_list, f, ensure_ascii=False, indent=4)

    qjudge_counter = Counter(item['Qjudge'] for item in cq_list)
    print("Qjudge 分布情况：")
    for key in ["0", "1"]:
        print(f"Qjudge '{key}': {qjudge_counter.get(key, 0)}")

if __name__ == "__main__":
    configs = argparser()
    process()
