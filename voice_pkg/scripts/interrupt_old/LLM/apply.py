import os
import yaml

from ..utils import read_yaml_from_parent

from .testgpt4 import gpt
from .testglm import glm_stream

def _llm_api(query=None, model='gpt-4'):
    if model == 'gpt-4' or model == 'gpt-3.5-turbo':
        reslist = gpt(model=model, text=query)
    elif model == 'chatglm':
        reslist = glm_stream(query=query)
    else:
        assert("The model you entered is not supported, please select the following model: gpt-4 gpt-3.5-turbo chatglm")
    result = ''
    for r in reslist:
        if r:
            result += str(r)
    return result

def text_correct(text=None, model='gpt-4'):
    query_prefix =  "我会给你提供一段航天馆游客向讲解员说的话（注意，是游客所说的话，而不是讲解员所说的话）。如果我提供的文本与“航天”主题相关性很小，请直接输出原文本。如果我提供的文本意图清晰、语意连贯，就直接输出原文本。如果我提供的文本意图不清淅或语义不连贯，请你对文本进行改写，使改写后的文本意图清晰、语义连贯、语言极其简洁没有任何废话、要符合游客身份。例如：化工大改为哈工大、博物馆改为航天馆、航空改为航天等等。注意！如果原文本中含有血腥暴力、歧视贬低等不符合规范的内容，请你只输出一个单独的X。你的输出只有两种选择：只输出改写后的文本或只输出X，一定不要输出其他内容。需要你处理的原文本："
    query_suffix = ""
    
    query = query_prefix + text + query_suffix

    # print('\nLLM rewrite prompt: \n', query, '\n')

    result = _llm_api(query=query, model=model)

    return result

def _prompt_construct(txt = "我想看看东方红卫星"):
    scenset = ("你是哈尔滨工业大学航天馆的展厅机器人，负责外来游客的导游任务。展厅有卫星展厅和火箭展厅。\n"
    "卫星展厅里有东方红卫星、紫丁香一号卫星等，火箭展厅包含了火箭一号，神州十五号等。\n")
    maintask = "你的主线任务是按照预定好的目标带游客按顺序参观航天馆的所有展厅包括卫星展厅和火箭展厅，同时游客可能随时打断你，你需要对游客的意图进行识别并分类。\n"
    taskset = ("你需要对游客的文本指令进行意图识别，根据游客的意图进行自身行为的规划，类别包含了 参观，休眠，问答，继续。\n"
    "当你识别到游客明确想要参观某个展厅时，比如游客说“带我看看火箭展厅吧”或“我想看看紫丁香卫星”，你需要将任务分类到“参观”，进而终止当前的主线任务，导航到游客指定的展厅，并输出‘参观 卫星展厅’或‘参观 火箭展厅’; "
    "当你识别到游客想了解问题时，比如游客说“我想了解一下北斗卫星”，你需要将任务分类到“问答”，进而执行问答任务，并输出‘问答’;"
    "当你识别到游客想要打断你正在执行的任务时，比如“可以了停下吧”或“我不想听了”，你需要将任务分类到“休眠”，进而停止你所有正在执行的任务，进入等待状态，并输出‘休眠’；"
    "当你识别到游客想让你继续主线任务时，比如“我没有问题了”或“没有了”或“继续参观吧”，你需要将任务分类到继续，从而继续执行主线任务，并输出‘继续’。\n"
    "如果识别到的指令与自身任务相关性很小，比如‘U盘’，则输出‘无关’\n"
    "以下是游客的指令，请你对其进行意图分类：")

    prompt = scenset + maintask + taskset + txt

    return prompt

def _prompt_construct_by_config(txt="我想看看东方红卫星", config_path="config.yaml"):
    # 读取配置文件
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # 构建展厅和展品信息
    sceneset = "你是哈尔滨工业大学航天馆的展厅机器人，负责外来游客的导游任务。展厅有" + \
               "和".join([museum['name'] for museum in config['museums']]) + "。\n" + \
               "".join([f"{museum['name']}里有{', '.join(museum['exhibits'])}等，" for museum in config['museums']]).rstrip('，') + "。\n"
    maintask = "你的主线任务是按照预定好的目标带游客按顺序参观航天馆的所有展厅包括" + \
               "和".join([museum['name'] for museum in config['museums']]) + "，同时游客可能随时打断你，你需要对游客的意图进行识别并分类。\n"
    
    # 构造few shot部分
    tasks_list = config['task_categories']
    few_shots = {
        "参观": "当你识别到游客明确想要参观某个展厅时，比如游客说“{}”，你需要将任务分类到“参观”，进而终止当前的主线任务，导航到游客指定的展厅，并输出‘visit 卫星展厅’或‘visit 火箭展厅’; ".format("'或'".join(config['few_shots']['参观'])),
        "问答": "当你识别到游客想了解问题时，比如游客说“{}”，你需要将任务分类到“问答”，进而执行问答任务，并输出‘qa’;".format("'或'".join(config['few_shots']['问答'])),
        "休眠": "当你识别到游客想要打断你正在执行的任务时，比如“{}”，你需要将任务分类到“休眠”，进而停止你所有正在执行的任务，进入等待状态，并输出‘sleep’；".format("'或'".join(config['few_shots']['休眠'])),
        "继续": "当你识别到游客想让你继续主线任务时，比如“{}”，你需要将任务分类到继续，从而继续执行主线任务，并输出‘continue’。".format("'或'".join(config['few_shots']['继续']))
    }

    taskset = "".join(few_shots.values())

    prompt = sceneset + maintask + taskset + "以下是游客的指令，请你对其进行意图分类：" + txt

    return prompt

def sleep_judge(text=None, model='gpt-4', mode='llm'):
    prompt_version = 'songhang'

    if mode == 're':
        query_prefix =  "我会给你提供一段航天馆游客向智能导游机器人说的话（注意，是游客所说的话，而不是讲解员所说的话），请你对提供的文本进行意图理解，判断游客是否想让机器人去休眠，也就是说游客不想让机器人讲解了，想让机器人停止。游客说的话是："
        query_suffix = "。你的输出只有两种选择：如果游客想让机器人去休眠，请你输出是；如果游客不想让机器人去休眠，请你输出否。不要输出这两种选择之外的内容！"
        query = query_prefix + text + query_suffix

    elif mode == 'llm':
        # 在代码中写死的prompt
        # query = _prompt_construct(text)

        # 根据配置文件灵活改写的prompt
        config_path = read_yaml_from_parent(config_filename='config.yaml', parent_levels=1)
        query = _prompt_construct_by_config(txt=text, config_path=config_path)

    # print('\nLLM task classification prompt: \n', query, '\n')

    result = _llm_api(query=query, model=model)

    # print('多分类: ', result)
    return result