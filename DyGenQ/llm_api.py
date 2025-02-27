import os
from dashscope import Generation  # 导入Qwen的API库
from openai import OpenAI
from volcenginesdkarkruntime import Ark


# GPT API 配置信息
client_gpt = OpenAI(
    api_key="xxx",
    base_url="https://aihubmix.com/v1"
)

# Qwen 配置
os.environ["DASHSCOPE_API_KEY"] = "xxx"

# Doub API 配置信息
os.environ["ARK_API_KEY"] = "xxx"
client_doub = Ark(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

# 定义各个 API 调用函数
def call_qwen_api(prompt):
    """
    调用 Qwen API 获取结果。
    """
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]
    response = Generation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),  # 确保环境变量已经配置
        model="qwen-plus",
        messages=messages,
        result_format="message"
    )
    if response.status_code == 200:
        return response.output.choices[0].message.content.strip()  # 去掉首尾空格
    else:
        raise RuntimeError(f"Qwen API 请求失败: {response.message}")

# 定义各个 API 调用函数 联网
def call_qwen_net_api(prompt):
    """
    调用 Qwen API 获取结果。
    """
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]
    response = Generation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),  # 确保环境变量已经配置
        model="qwen-plus",
        messages=messages,
        result_format="message",
        enable_search=True
    )
    if response.status_code == 200:
        return response.output.choices[0].message.content.strip()  # 去掉首尾空格
    else:
        raise RuntimeError(f"Qwen API 请求失败: {response.message}")

def call_gpt3_api(prompt):
    """
    调用 gpt-3.5-turbo API 获取结果。
    """
    chat_completion = client_gpt.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message.content

def call_gpt4_api(prompt):
    """
    调用 GPT-4 API 获取结果。
    """
    chat_completion = client_gpt.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o-mini",
    )
    return chat_completion.choices[0].message.content

def call_claude3_api(prompt):
    """
    调用 claude3 API 获取结果。
    """
    chat_completion = client_gpt.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="claude-3-5-haiku-20241022",
        
    )
    return chat_completion.choices[0].message.content

def call_glm4_api(prompt):
    """
    调用 glm-4-flash API 获取结果。
    """
    chat_completion = client_gpt.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="glm-4-flash",
    )
    return chat_completion.choices[0].message.content

def call_deepseek_api(prompt):
    """
    调用 deepseek-reasoner API 获取结果。
    """
    chat_completion = client_gpt.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        # model="deepseek-ai/DeepSeek-V3",
        model="deepseek-ai/DeepSeek-V3",
    )
    return chat_completion.choices[0].message.content


def call_llama3_api(prompt):
    """
    调用 llama-3.3-70b API 获取结果。
    """
    chat_completion = client_gpt.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="aihubmix-Llama-3-3-70B-Instruct",
    )
    return chat_completion.choices[0].message.content


def call_doub_api(prompt):
    """
    调用 Doub API 获取结果。
    """
    completion = client_doub.chat.completions.create(
        model="ep-20241226085514-qx4cx",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return completion.choices[0].message.content

