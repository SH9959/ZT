"""openai.APIConnectionError: Connection error"""
import os
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
"""openai.APIConnectionError: Connection error"""

import time
import yaml
from openai import OpenAI

from ..utils import read_yaml_from_parent

# Example of an OpenAI ChatCompletion request
# https://platform.openai.com/docs/guides/text-generation/chat-completions-api

def get_gpt4_key():
    # 读取openai api key
    config_path = read_yaml_from_parent(config_filename='prompt_config.yaml', parent_levels=1)
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    openai_api_key = config['openai_api_key'][0]
    return openai_api_key

def gpt(model:str, text:str):
    open_api_key = get_gpt4_key()
    client = OpenAI(api_key=open_api_key)

    # send a ChatCompletion request to count to 100
    response = client.chat.completions.create(
        model=model,  # 'gpt-3.5-turbo' 'gpt-4'
        messages=[
            {'role': 'user', 'content': text}
        ],
        temperature=0,
        stream=True  # again, we set stream=True
    )
    for chunk in response:
        yield chunk.choices[0].delta.content  # extract the message)

if __name__ == '__main__':

    reslist = gpt(model='gpt-4', text="你好")
    for r in reslist:
        if r:
            print(r, end='')