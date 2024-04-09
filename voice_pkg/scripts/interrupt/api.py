from .PPL import get_ppl_for
from .LLM import text_correct, user_ensure, DocumentQAGPT4, DocumentQAHuoZi

def get_ppl_bool(text=None, threshold=100):
    """
    输入一段文本，输出文本的PPL值
    param text: 待检测的文本
    param threshold: 判断流畅性是否足够的阈值
    """
    ppl = get_ppl_for(sentence=text)
    if ppl[0] <= threshold:
        return True
    else:
        return False
    
def get_llm_rewrite(text=None, model='gpt-4'):
    """
    输入一段文本，输出流畅性、相关性增强改写后的文本，如果是毒性内容则返回X
    param text: 待改写的文本
    param model: 大模型类型，候选 'gpt-4' 'chatglm' 'gpt-3.5-turbo'
    """
    return text_correct(text=text, model=model)

def get_task_type(text=None, keyword_list=None, model='gpt-4', mode='llm'):
    """
    输入一段文本，返回任务类型（导航，继续，睡眠，问答）
    param text: 待判断的文本
    param keyword_list: 导航目标点（各种展厅）列表
    param model: 大模型类型，候选 'gpt-4' 'chatglm' 'gpt-3.5-turbo'
    param mode: 识别任务类型的方案，正则或大模型，候选 're' 'llm'
    """
    from .CLS import analyze_text
    return analyze_text(text=text, keyword_list=keyword_list, model=model, mode=mode)

def get_llm_answer(model='huozi'):
    """
    输入模型类型，返回一个支持多轮问答的模型类，通过 doc_qa_class.process_query("问题示例") 来调用
    param text: 待判断的文本
    param model: 大模型类型，候选 'gpt-4' 'huozi'
    """
    if model == 'huozi':
        doc_qa_class = DocumentQAHuoZi
    elif model == 'gpt-4':
        doc_qa_class = DocumentQAGPT4

    # doc_qa_class.process_query("介绍一下北斗卫星")
    return doc_qa_class

def get_llm_check(text=None, model='gpt-4'):
    """
    输入一段待确认的文本，返回值可能是True Flase 用户重新给出的文本
    param text: 待改写的文本
    param model: 大模型类型，候选 'gpt-4' 'chatglm' 'gpt-3.5-turbo'
    """
    return user_ensure(text=text, model=model)
    

if __name__ == '__main__':
    same_text = "别跟着我"

    # print(get_ppl_bool(text=same_text, threshold=100))
    # print(get_llm_rewrite(text=same_text, model='gpt-4'))
    # print(get_task_type(text='带我去卫星展厅', keyword_list=['卫星展厅', '火箭展厅'], model='gpt-4'))
    chat_model = get_llm_answer(model='huozi')
    print(chat_model.process_query("介绍一下哈工大航天馆"))
