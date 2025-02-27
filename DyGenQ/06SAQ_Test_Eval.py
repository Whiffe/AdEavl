'''
python 06SAQ_Test_Eval.py \
    --promptTest ./prompt/SAQTest.txt \
    --promptEval ./prompt/SAQEval.txt \
    --dataset ./data/SAQ.json \
    --output ./output/SAQ_Answer.json

    
python 06SAQ_Test_Eval.py \
    --promptTest ./prompt/SAQTest.txt \
    --promptEval ./prompt/SAQEval.txt \
    --dataset ./data/SAQ_format.json \
    --output ./output/SAQ_Answer_SAQ_format.json
'''
from llm_api import qwen_7B, call_qwen_api
import argparse
import json
from tqdm import tqdm  # 导入 tqdm 库
import os

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--promptTest', default='./prompt/SAQTest.txt', type=str)
    parser.add_argument('--promptEval', default='./prompt/SAQEval.txt', type=str)
    parser.add_argument('--dataset', default='./output/SAQ_format.json', type=str)
    parser.add_argument('--output', default='./output/SAQ_Answer.json', type=str)
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
                promptTest = read_txt(configs.promptTest).replace("{{SAQ}}", row['instruction'])
                resTest = qwen_7B(promptTest)  # 调用小模型 API
                
                promptEval = read_txt(configs.promptEval).replace("{{SAQ}}", row['instruction']).replace("{{SAQ_S_Answer}}", row['output']).replace("{{SAQ_Answer}}", resTest)
                resEval = call_qwen_api(promptEval)  # 调用大模型 API
                
                saq_item = {
                    "instruction": row['instruction'],
                    "output": row['output'],
                    "answer": resTest,
                    "score": resEval
                }

                saq_list.append(saq_item)
            
                # 更新进度条
                pbar.update(1)
                
                # 将 SAQ 数据写入 JSON 文件
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(saq_list, f, ensure_ascii=False, indent=4)
                    
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请检查路径是否正确。")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")   
if __name__ == "__main__":
    configs = argparser()  # 加载参数配置
    process(configs.dataset, configs.output)