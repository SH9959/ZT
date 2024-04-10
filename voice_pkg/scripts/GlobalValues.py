# ===============
# Author: hsong
# Quote: 先不用property封装
# ===============


from typing import Dict, List, Tuple, Optional
import inspect
import yaml, os
from copy import deepcopy

from main.msg import actionStopper

class GlobalValuesClass:
    """
    Class Description: 用于管理全局数值的类。

    Attributes:
        _id (int): 类属性，用于记录实例的唯一标识符。
        _all_instances (dict): 类属性，用于存储所有实例的字典，键为实例ID，值为实例对象。
        
    到达一个地方，需要维护：
        STATUS.set_Current_Area(new_Current_Area)
    """

    _id = 0 
    _all_instances = {}  

    def __init__(                                           #                                * 表示重要程度
            self, 
            
            TAKE_ACTION:bool=False,                         # 是否开启机器人做动作功能
            FACE_DETECT:bool=False,                         # 是否开启人脸检测功能
            OBSTAC_STOP:bool=False,                         # 是否开启停障功能
            POSE_DETECT:bool=False,                         # 是否开启姿势检测功能

            MODEL_TASK_TYPE:str = "",                       # 用于做任务分类的模型
            MODEL_LLM_ANSWER:str = "",                      # 用于回答问题的模型
            MODEL_BAN_OPENAI:str = "",                      # 是否禁用OpenAI
            
            info: Optional[str] = None,                     # 当前实例的描述信息，随意
            name:str="This is a glabolvalues",              # 当前实例的描述信息，随意
            is_Navigating:bool=False,                       # robot是否处于移动状态            *
            is_Depth_Obstacle=False,                        # robot是否从深度信息检测到障碍物
            is_Yolo_Obstacle=False,                         # robot是否从yolo信息检测到障碍物
            is_Explaining:bool=False,                       # robot是否处于讲解状态            *
            is_QAing:bool=False,                            # robot是否处于QA状态              *
            Action_from_User:str="None",                    # 用户希望的动作
            is_Interrupted:bool=False,                      # robot是否处于被打断的状态        **
            is_Big_Face_Detected:bool=False,                # robot是否检测到人脸              **
            Big_Face_Area:str = "LEFT",                     # robot检测到的人脸区域
            
            Block_Navigation_Thread:bool=False,             # 是否阻止导航进程返回
            Current_Area:str="火箭展厅",                     # robot当前area                   *
            Destination_Area:str="火箭展厅",                 # Navigate目标area                **
            Current_Position:Optional[List[float]]=None,    # robot当前精确坐标，可不写       
            Target_Position:Optional[List[float]]=None,     # robot目标精确坐标，可不写
            is_Arrived:bool=False,                          # robot到达指定地点，
            
            Index_of_Document:int = 0,                      # 讲解被打断时 记录的文稿索引位置    *
            Last_Sentence:str = "",                         # 讲解被打断时 记录的最近一句话      *
            
            Interrupt_Area:Optional[str]=None,              # 移动被打断时 记录当时所处的展厅区域 *
            Interrupt_Position:Optional[List[float]]=None,  # 移动被打断时 记录当时所处的精确坐标

            Stop_Publisher:Optional[actionStopper]=None,    # 停止发布者
            
            Origin_Order_of_Visit:List[str]=[               # robot默认原始参观顺序列表
                "火箭展厅", 
                "卫星展厅", 
                "航天器展厅", 
                "运载火箭展厅"
                ],
            Current_Order_of_Visit:List[str]=[               # robot当前任务列表，完成一个就pop一个
                "火箭展厅", 
                "卫星展厅", 
                "航天器展厅", 
                "运载火箭展厅"
                ],
            Mask:List[bool] = [False] * 4,                                 # 到过的地方就标记
            Onehot:List[bool] = [False] * 4,                               # 现在位于哪里
            Last_Play_Processor = None,
            
        ):

        self.id = GlobalValuesClass._id
        GlobalValuesClass._all_instances[self.id] = self  
        GlobalValuesClass._id += 1

        self.TAKE_ACTION = TAKE_ACTION
        self.FACE_DETECT = FACE_DETECT
        self.OBSTAC_STOP = OBSTAC_STOP
        self.POSE_DETECT = POSE_DETECT

        self.MODEL_TASK_TYPE = MODEL_TASK_TYPE
        self.MODEL_LLM_ANSWER = MODEL_LLM_ANSWER
        self.MODEL_BAN_OPENAI = MODEL_BAN_OPENAI
        
        if info is None:
            info = f"Hello! Description here." 
        self.info =  f"{self.__class__.__name__}"+ " " + name + ": " + info # 描述  
        
        self.is_Navigating = is_Navigating
        self.is_Depth_Obstacle = is_Depth_Obstacle
        self.is_Yolo_Obstacle = is_Yolo_Obstacle
        self.is_Explaining = is_Explaining
        self.is_QAing = is_QAing
        self.Action_from_User = Action_from_User
        self.is_Interrupted = is_Interrupted
        self.is_Big_Face_Detected = is_Big_Face_Detected
        self.Big_Face_Area = Big_Face_Area
        
        self.Block_Navigation_Thread = Block_Navigation_Thread
        self.Current_Area = Current_Area     
        
        self.Destination_Area = Destination_Area
        self.Current_Position = Current_Position
        self.Target_Position = Target_Position
        self.Index_of_Document = Index_of_Document
        self.Last_Sentence = Last_Sentence
        self.Interrupt_Area = Interrupt_Area
        self.Interrupt_Position = Interrupt_Position
        self.Last_Play_Processor = Last_Play_Processor

        self.Stop_Publisher = Stop_Publisher

        if self.Current_Area:
            self.Last_Area = self.Current_Area                 # 记录上一次呆过的地方
        else:
            self.Last_Area = ""

        config_path = 'config/config.yaml'
        if not os.path.exists(config_path):
            self.Origin_Order_of_Visit = Origin_Order_of_Visit
            self.Current_Order_of_Visit = Current_Order_of_Visit
        else:
            with open(config_path, 'r') as file:
                data = yaml.safe_load(file)
            #print(data)
            self.Origin_Order_of_Visit = list(data['讲解词列表'].keys())
            self.Current_Order_of_Visit = list(data['讲解词列表'].keys())

        self.Mask = [False for _ in self.Origin_Order_of_Visit]
        self.Onehot = [False for _ in self.Origin_Order_of_Visit]

    def get_first_Current_Order_of_Visit_id(self, ):
        return self.Origin_Order_of_Visit.index(self.Current_Order_of_Visit[0]) if len(self.Current_Order_of_Visit) > 0 else None
        
    def set_TAKE_ACTION(self, new_TAKE_ACTION:bool=False) -> None:
        self.TAKE_ACTION = new_TAKE_ACTION

    def set_FACE_DETECT(self, new_FACE_DETECT:bool=False) -> None:
        self.FACE_DETECT = new_FACE_DETECT

    def set_OBSTAC_STOP(self, new_OBSTAC_STOP:bool=False) -> None:
        self.OBSTAC_STOP = new_OBSTAC_STOP

    def set_POSE_DETECT(self, new_POSE_DETECT:bool=False) -> None:
        self.POSE_DETECT = new_POSE_DETECT

    def set_is_Navigating(self, new_is_Navigating:bool=False) -> None:
        self.is_Navigating = new_is_Navigating
        if new_is_Navigating:
            self.is_Arrived = False
            
    def set_is_Depth_Obstacle(self, new_is_Depth_Obstacle:bool=False) -> None:
        self.is_Depth_Obstacle = new_is_Depth_Obstacle

    def set_is_Yolo_Obstacle(self, new_is_Yolo_Obstacle:bool=False) -> None:
        self.is_Yolo_Obstacle = new_is_Yolo_Obstacle
        
    def set_is_Explaining(self, new_is_Explaining:bool=False) -> None:
        self.is_Explaining = new_is_Explaining
        self.is_Navigating = False
        
    def set_is_QAing(self, new_is_QAing:bool=False) -> None:
        self.is_QAing = new_is_QAing
        
    def set_Action_from_User(self, new_Action_from_User:str="None") -> None:
        self.Action_from_User = new_Action_from_User

    def set_is_Interrupted(self, new_is_Interrupted:bool=False):
        self.is_Interrupted = new_is_Interrupted

    def set_is_Big_Face_Detected(self, new_is_Big_Face_Detected:bool=False) -> None:
        self.is_Big_Face_Detected = new_is_Big_Face_Detected
        
    def set_Big_Face_Area(self, new_Big_Face_Area:str="CENTER") -> None:
        self.Big_Face_Area = new_Big_Face_Area

    def set_Block_Navigation_Thread(self, new_Block_Navigation_Thread:bool=False)-> None:
        self.Block_Navigation_Thread = new_Block_Navigation_Thread

    def set_Current_Area(self, new_Current_Area:str="火箭展厅") -> None:
        self.Current_Area = new_Current_Area
        self.Mask[self.Origin_Order_of_Visit.index(self.Current_Area)] = True
        # self.Onehot[self.Origin_Order_of_Visit.index(self.Current_Area)] = True
        for i, area in enumerate(self.Origin_Order_of_Visit):
            self.Onehot[i] = area == self.Current_Area
        self.Last_Area = new_Current_Area
        
        if self.Destination_Area == self.Current_Area:
            self.is_Arrived = True
            self.set_is_Navigating(False)
        
    def set_Last_Area(self, new_Last_Area:str):
        self.Last_Area = new_Last_Area
        
    def set_is_Arrived(self, new_is_Arrived:bool):
        self.is_Arrived = new_is_Arrived
        if self.is_Arrived:
            self.is_Navigating = False
        
    def set_Destination_Area(self, new_Destination_Area:str="航天器展厅") -> None:
        if new_Destination_Area not in self.Origin_Order_of_Visit:
            print(f"\033[33mWarning\033[0m {new_Destination_Area} 不在范围内")
        def get_new_order(POLICY: str='SKIP') -> List[int]:
            if POLICY == "SKIP":  # 升序策略
                index = self.Origin_Order_of_Visit.index(new_Destination_Area)
                new_list = deepcopy(self.Origin_Order_of_Visit[index:])
                print("\nnew_list", new_list)
            return new_list
                
    
        # if self.Destination_Area != new_Destination_Area:  # 重构list
        self.Current_Order_of_Visit = get_new_order(POLICY="SKIP")
        # print(self.Current_Order_of_Visit)
            
        self.Destination_Area = new_Destination_Area
        
        if self.Destination_Area != self.Current_Area:  # 未到
            self.is_Arrived = False
        else:
            self.is_Arrived = True

    def set_Current_Position(self, new_Current_Position:Optional[List[float]]) -> None:
        self.Current_Position = new_Current_Position

    def set_Target_Position(self, new_Target_Position:Optional[List[float]]) -> None:
        self.Target_Position = new_Target_Position

    def set_Index_of_Document(self, new_Index_of_Document:int = 0) -> None:
        self.Index_of_Document = new_Index_of_Document

    def set_Last_Sentence(self, new_Last_Sentence:str="") -> None:
        self.Last_Sentence = new_Last_Sentence

    def set_Interrupt_Area(self, new_Interrupt_Area:str="Gate") -> None:
        self.Interrupt_Area = new_Interrupt_Area

    def set_Interrupt_Position(self, new_Interrupt_Position:Optional[List[float]]) -> None:
        self.Interrupt_Position = new_Interrupt_Position

    def set_Last_Play_Processor(self, new_Last_Play_Processor):
        self.Last_Play_Processor = new_Last_Play_Processor

    def set_Stop_Publisher(self, new_Stop_Publisher):
        self.Stop_Publisher = new_Stop_Publisher
        
    def update_all_attributes(self, new_attributes: dict) -> None:
        """
        通过字典方式更新所有实例变量的值。

        Args:
            new_attributes (dict): 包含要更新的实例变量及其新值的字典。

        """
        for attr, value in new_attributes.items():
            if hasattr(self, attr):
                if attr == "id":
                    print("Warning: The 'id' is not allowed to modified. Skip.")
                    continue
                setattr(self, attr, value)
            else:
                print(f"\nWarning: Attribute '{attr}' does not exist in {self.__class__.__name__}.\n")
    
    
    def add_var(self, var_name, value):
        """添加临时实例变量

        """
        setattr(self, var_name, value)
        
    def del_var(self, var_name):
        """删除临时实例变量

        """
        if hasattr(self, var_name):  # 检查实例是否包含指定的属性
            delattr(self, var_name)  # 删除实例变量
            
    def get_states_dict(self, ):
        return self.__dict__
            
    @classmethod
    def get_instance_by_id(cls, id):  # 根据ID获取实例
        return cls._all_instances.get(id, None)  

    @classmethod
    def get_count(cls):
        return cls._id

    @classmethod
    def create_instance(cls,**kwargs):   #  可以不用
        return cls(**kwargs)  # 返回当前类的实例
    
if __name__ == "__main__":
#    a = NavigationClass()
#     print(type(a.get_self_name()))

    STATUS = GlobalValuesClass(name="STATUS_1")
    STATUS_DICT = STATUS.get_states_dict()
    print(STATUS_DICT)
    
    print(STATUS.is_Explaining)
    print(STATUS_DICT['is_Explaining'])
    
    STATUS.is_Explaining = True
    print(STATUS.is_Explaining)
    print(STATUS_DICT['is_Explaining'])
    
    STATUS_DICT['is_Explaining']= False
    print(STATUS.is_Explaining)
    print(STATUS_DICT['is_Explaining'])
    
    STATUS.set_Destination_Area(new_Destination_Area="火箭展厅")
    
    STATUS.get_first_Current_Order_of_Visit_id()
    
    # a1 = GlobalValuesClass(name="a1")
    # a2 = GlobalValuesClass.create_instance(name="jaha")
    
    # a1.set_is_Explaining(False)  # 更新方式1
    # print(a1)
    
    # print(vars(a1))
    # print(vars(a2))
    # print(a2.__dict__)
    
    
    
    # aa = {'id': 100, 'is_Explaining': True, 'bb': '展厅', 'c': {'asd': 9}, 'e': ('asd', 'asda', 'er'), 'd': [1, 2, 3]}
    
    # T = {
    #     'id': 1, 
    #     'info': 'GlobalValuesClass jaha: Hello! Description here.', 
    #     'is_Navigating': False, 
    #     'is_Explaining': True, 
    #     'is_QAing': False, 
    #     'Current_Area': 'Gate', 
    #     'Destination_Area': '卫星展厅', 
    #     'Current_Position': None, 
    #     'Target_Position': None, 
    #     'Index_of_Document': 0, 
    #     'Last_Sentence': '', 
    #     'Interrupt_Area': None, 
    #     'Interrupt_Position': None
    #     }
    # a2.update_all_attributes(T)  # 更新方式2
    # print(a2.__dict__)
    # pass