from sentence_transformers import SentenceTransformer
import time
import numpy as np

# _model = SentenceTransformer('BAAI/bge-large-zh-v1.5')  # 1.3G
# _model = SentenceTransformer('BAAI/bge-base-zh-v1.5')   # 409M
# _model = SentenceTransformer('BAAI/bge-small-zh-v1.5')  # 100M-200M
# 效果递减
_model = SentenceTransformer('checkpoints/qa_bge_small_zh_v15')

def get_action_from(txt:str, _model:object=_model):
    """给定一段文本，匹配合适的动作
    
    Args:
        txt (str): 输入文本
        _model (object, optional): 模型. Defaults to _model.
    Returns:
        str: 匹配到的动作
        None: 无匹配
    
    """
    actions = ["招手", "举左手", "举右手","向前指","向上指","向后转", "摊手", "指向我自己"]

    embeddings_1 = _model.encode(txt, normalize_embeddings=True)
    # print(embeddings_1)
    embeddings_2 = _model.encode(actions, normalize_embeddings=True)
    # print(embeddings_2)
    similarity = embeddings_1 @ embeddings_2.T
    print(f"输入句子：{txt}:")
    print(similarity)
    print(actions)
    
    argmax = np.argmax(similarity)
    #print(argmax)
    if similarity[argmax] > 0.4:
        print("输出动作：",actions[argmax])
        print(" ")
        return actions[argmax]
    else:
        print('没有适合的动作')
        print(" ")
        return None




if __name__=="__main__":

    txt = [
        "你好", 
        "再见",
        "请看左边这个展品",
        "请看右边这个展品", 
        "请看前方这个展品", 
        "请看上面这个展品",
        "我们",
        "请看后面这个展品",
        "给大家介绍一下",
        "介绍一下风云4号"
        ]
    for i in txt:
        action = get_action_from(i)

