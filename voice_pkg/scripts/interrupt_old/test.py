from .api import get_ppl_bool
from .api import get_llm_rewrite
from .api import get_task_type

if __name__ == '__main__':
    model = 'gpt-4'

    # text = "我想看看东方红"
    # text = "带我继续参观"
    # text = "我没问题了"
    # text = "我想参观卫星展厅"
    # text = "请继续讲解"
    # text = "我不需要你讲解了"
    # text = "给我讲讲东方红卫星"
    # text = "东方红卫星的上市时间"
    # text = "皮卡丘"
    text = "卫星展厅里有意思吗"

    # Step 1: PPL计算
    ppl_bool = get_ppl_bool(text=text, threshold=150)
    print(ppl_bool)

    if 1:
        # Step 2: 大模型改写
        rewrite_result = get_llm_rewrite(text=text, model=model)
        print('大模型改写结果: ', rewrite_result)

        # Step 3: 判断任务类型
        print('任务类型: ', get_task_type(text=rewrite_result, keyword_list=['卫星展厅', '火箭展厅'], model=model, mode='llm'))