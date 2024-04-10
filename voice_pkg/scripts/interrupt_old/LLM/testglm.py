"""openai.APIConnectionError: Connection error"""
import os
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
"""openai.APIConnectionError: Connection error"""

import json
from zhipuai import ZhipuAI

def get_zhipu_key():
    with open('/home/kuavo/catkin_dt/config_dt.json', 'r') as fj:
        config = json.load(fj)
    zhipu_api_key = config['zhipu_apikey']
    return zhipu_api_key

def glm_stream(query, system_prompt=None, few_shots=None, debug=False):
    zhipu_api_key = get_zhipu_key()
    client = ZhipuAI(api_key=zhipu_api_key) # 填写您自己的APIKey

    if system_prompt is None:
        system_prompt = ""
    assert type(system_prompt) is str
    if few_shots is None:
        few_shots = []
    assert type(few_shots) is list

    content = system_prompt

    for i in range(len(few_shots)):
        content += f"输入：{few_shots[i][0]}\n"
        content += f"输出：{few_shots[i][1]}\n"

    content += f"输入：{query}\n输出："

    conversations = [{"role": "user", "content": content}]
    
    response = client.chat.completions.create(
        model="glm-3-turbo", # 填写需要调用的模型名称
        messages=conversations,
        stream=True
    )
    # return response.choices[0].message.content
    for resp in response:
        yield resp.choices[0].delta.content

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

        reslist = glm_stream(query=query)
        for r in reslist:
            print(r, end='')
        print("\n")

if __name__ == 'main':
    # text_correct()

    reslist = glm_stream(text="你好")
    for r in reslist:
        print(r, end='')
