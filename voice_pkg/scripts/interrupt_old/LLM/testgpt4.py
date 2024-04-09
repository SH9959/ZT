"""openai.APIConnectionError: Connection error"""
import os
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
"""openai.APIConnectionError: Connection error"""

import time
from openai import OpenAI

# Example of an OpenAI ChatCompletion request
# https://platform.openai.com/docs/guides/text-generation/chat-completions-api

def gpt(model:str, text:str):
    open_api_key = "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"
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

def text_correct():
    query_prefix =  "我会给你提供一段航天馆游客向讲解员说的话（注意，是游客所说的话，而不是讲解员所说的话），请你对提供的文本进行深度改写（即不需要过于依赖源文本，而是更注重语意连贯和文本与“航天”的相关性），使改写后的文本语义连贯、语言极其简洁没有任何废话、要符合游客身份。例如，例如，化工大改为哈工大、博物馆改为航天馆、航空改为航天等等。需要你处理的原文本："
    query_suffix = "。请只输出改写后的文本，不要输出多余的内容。"

    texts = ["东方红卫星何时上市",
            "介绍一下哈工大在航空领域的专业服务",
            "介绍一下哈工大在航天的一份科研成果",
            "2手展厅展示的是什么",
            "哈工大博物馆成立于什么时候",
            "介绍一下哈，工大附中在中国",
            "他们的还不馆成立于什么时候起",
            "介绍他并的卫星"]

    print("\n")

    for text in texts:
        print(text, " ->  ", end="")

        query = query_prefix + text + query_suffix

        # print(query)
        # exit()

        reslist = gpt(model='gpt-4', text=query)
        for r in reslist:
            print(r, end='')
        print("\n")

if __name__ == '__main__':
    # text_correct()

    reslist = gpt(model='gpt-4', text="你好")
    for r in reslist:
        print(r, end='')