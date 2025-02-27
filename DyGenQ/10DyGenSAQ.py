''''
一个问题生成多个，步骤一步步变得复杂
第一步是，把选择题变成简答题
'''
from llm_api import call_qwen_api, call_qwen_net_api  # 假设你使用这个 API 函数
import argparse
import json
from tqdm import tqdm  # 导入 tqdm 库
import os

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='./prompt/choice2SAQ.txt', type=str)
    parser.add_argument('--dataset', default='./dataset/cmmlu0.0125.json', type=str)
    parser.add_argument('--output', default='./output/SAQ.json', type=str)
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
    try:
        data = read_json(file_path)

        # 用于存储所有的 SAQ 数据
        saq_list = []

        # 使用 tqdm 创建进度条
        with tqdm(total=len(data), desc="Processing questions") as pbar:
            # 遍历数据并将每一行合成为一个字典
            for index, row in enumerate(data):
                question_data = {
                    "question": row["instruction"],  # 获取 question
                    "answer": row["output"]      # 获取 answer
                }
                choiceQ = f"问题：{question_data['question']} \n回答：{question_data['answer']}"
                # 重试逻辑，最多3次
                retry_count = 0
                while retry_count < 3:
                    prompt = read_txt(configs.prompt).replace("{{choiceQ}}", choiceQ)
                    res = call_qwen_api(prompt)  # 调用大模型 API
                    
                    try:
                        SAQ = eval(res)
                        if isinstance(SAQ, list) and len(SAQ) == 2:
                            # 将结果存入列表
                            saq_entry = {
                                "instruction": SAQ[0],
                                "input": "",
                                "output": SAQ[1],
                                "system": "",
                                "history": []
                            }
                            saq_list.append(saq_entry)
                            break  # 如果有效，则跳出循环
                        else:
                            print(f"警告: 输出无效 ({SAQ})，正在重新请求...")
                            retry_count += 1
                    except Exception as e:
                        print(f"警告: 输出异常 ({res})，错误信息: {e}，正在重新请求...")
                        retry_count += 1

                    if retry_count == 3:
                        print(f"警告: 超过3次重试，跳过当前项: {choiceQ}")

                # 更新进度条
                pbar.update(1)

                # 将 SAQ 数据写入 JSON 文件
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(saq_list, f, ensure_ascii=False, indent=4)
        print(f"SAQ 数据已成功保存到 {output_path}")

    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请检查路径是否正确。")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

if __name__ == "__main__":
    configs = argparser()  # 加载参数配置
    process(configs.dataset, configs.output)
