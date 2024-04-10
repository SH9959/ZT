import time
import yaml
import inspect
import threading
import subprocess
import _thread as thread
from time import sleep
from typing import Dict, List, Tuple, Optional

from kedaxunfei_tts.tts_as_service import tts_playsound
from kedaxunfei_iat.iat_as_service_iter import iat_web_api
# from tencentclound.tencentclound import iat_tencent as iat_web_api

import rospy
from main.srv import msgCommand
from main.msg import actionStopper
from std_msgs.msg import String, Bool

from interrupt import get_task_type
from interrupt import get_llm_answer
from GlobalValues import GlobalValuesClass
# from Only_text_prompt.Robot_prompt_text import ask  #lhkong动作分类

STATUS = GlobalValuesClass(name="Anythin is OK")
STATUS_DICT = STATUS.get_states_dict()

# print(STATUS.is_Explaining)
# print(STATUS_DICT)
    
def text2speech(text='', index=0):
    '''
    需要成员变量：
    interrupt:记录是否打断
    lastplayproc:记录上一个 播放音频的 process
    '''
    ttsproc = subprocess.Popen(["python3", "/home/kuavo/catkin_dt/src/voice_pkg/scripts/kedaxunfei_tts/tts_as_service.py", text])
    while ttsproc.poll() is None:
        print('tts process is working, interrupted:', STATUS.is_Interrupted)
        if STATUS.is_Interrupted:
            ttsproc.kill()
            return
        time.sleep(0.5)
    
    while STATUS.Last_Play_Processor and STATUS.Last_Play_Processor.poll() is None:
        print('last process is working, waiting')
        if STATUS.is_Interrupted:
            STATUS.Last_Play_Processor.kill()
            return
        time.sleep(0.5)

    playproc = subprocess.Popen(["python3", "/home/kuavo/catkin_dt/src/voice_pkg/scripts/kedaxunfei_tts/playsound.py"])
    if index == 1000: 
        # 同步播放
        while playproc.poll() is None:
            print('play process is working')
            if STATUS.is_Interrupted:
                playproc.kill()
                return
            time.sleep(0.5)
        STATUS.Last_Play_Processor = None
    else:
        # 异步播放:
        STATUS.Last_Play_Processor = playproc
        # 等待的时间必不可少，因为会有playsound和tts的读写同一个文件的冲突，因此先playsound再让tts访问 play.wav
        time.sleep(0.15)
    return 'tts is over'

def listenuser(text='ding', iter=1):
    # 可选的录音保存路径：
    # % DJI_sitting_at_desk         % 坐在办公桌前，手持小蜜蜂
    # % DJI_stand_next_robot        % 站在机器人旁边（1-2米），手持小蜜蜂
    # % DJI_very_close_robot        % 小蜜蜂放在距离机器人风扇非常近的地方，比如放在机器人肩膀上
    # % HEAD_stand_next_robot       % 站在机器人旁边（1-2米），使用头部麦克风
    # % HEAD_very_close_robot       % 说话时非常贴近头部麦克风
    # 特殊情况添加以下后缀：
    # % _noisePeople                % 环境中有很多人声，非常吵
    # % _quiet                      % 周围基本没什么人说话，很安静
    userword = iat_web_api(text, iter=iter, environment_name='DJI_sitting_at_desk')

    return userword

def pardon(pardon_round=10):  # TODO:utils.
    # 反复录制用户语音，直至语音不为空，最多重复pardon_round轮
    print(f"开始录制用户语音")

    userword = listenuser('ding', iter=pardon_round)

    print(f"录制结束，录制结果为: {userword}")
    # userword = input()
    return userword


class InterruptClass:
    def __init__(self, qa_class:object):
        global STATUS
        
        self.qa_class = qa_class # 传入的问答类实例用于生成用户问题的回答

    def listen_for_INT(self, ) -> bool:
        """监听打断信号, 返回是否打断
        
        """
        return STATUS.is_Interrupted
        
    def handle_interrupt(self, ):
        """处理所有情况下用户打断的操作。

        """
        STATUS.set_is_QAing(True) # 防止问答被打断 - 开始
        STATUS.is_Interrupted = False # 由于已经在处理打断，故消除打断flag

        name = self.get_self_name() # 获取调用该方法的类的名称，例如NavigationClass, ExplainClass等
        print(f"\n处理 {name} 过程中的打断")

        # TODO: 机器人如何转向面朝用户？用户人脸居中才停下来，如果有多个人脸就匹配最大的

        if STATUS.FACE_DETECT:
            print("STATUS.is_Big_Face_Detected: ", STATUS.is_Big_Face_Detected)
            while not STATUS.is_Big_Face_Detected: # 视野中没有人脸
                # TODO: 向下位机发送转向命令
                # TODO: 如何判断向左还是向右？

                print("\n向下位机发送转向命令")
                rospy.wait_for_service('TurningService') # 等待与下位机通信的服务变得可用
                try:
                    turn = rospy.ServiceProxy('add_two_ints', AddTwoInts) # 创建服务代理，用于调用服务
                    responce = turn(int(2)) # 向服务发送请求，阻塞，直至到达目标点才会返回，正数左传，负数右转，最大15度
                except rospy.ServiceException as e:
                    print("Turn Service call failed: %s"%e)
            
            # 向下位机发送停止信号
            pub_string = actionStopper()
            pub_string.stopper = True
            STATUS.Stop_Publisher.publish(pub_string)
            print("\n因为转向看到人脸，向下位机发送停止信号")
            STATUS.is_Big_Face_Detected = False

        text2speech('大家有什么问题吗？', index=1000) # 提示用户说出问题
        
        pn_flag = True
        while pn_flag: # 该循环（称为PN）的唯一作用是：如果不存在'下一个'或'上一个'展厅，机器人会提示用户重新下达指令，此时程序会跳转到该循环的开始处
            pn_flag = False

            question = pardon() # 录制用户的指令
            print(f"\n用户指令: {question}")

            text2speech('好的。', index=1000) # 缓解 录音+任务分类 一大段空白
            
            # 启动线程用于识别用户希望机器人执行的动作
            if STATUS.TAKE_ACTION:
                thread.start_new_thread(self.ask_for_action, (question,))
            
            print(f"\n大模型进行任务分类: ", end='')
            task = get_task_type(question, model='chatglm') # 任务分类
            print(task)

            STATUS.Block_Navigation_Thread = False # 在消除打断flag之后延迟一些消除阻止导航进程的flag，保证导航进程被阻止
            
            # 一直进行问答，直到task不是QA，转而进行其他操作
            while "qa" in task:
                if not 'explicit' in task: # 如果问题中没有显示指出展品名称
                    pass

                self.qa_class.answer_question(question) # 生成答案并播音
                question = pardon() # 录制用户的指令
                print(f"\n用户指令: {question}")
                
                # 启动线程用于识别用户希望机器人执行的动作
                if STATUS.TAKE_ACTION:
                    self.ask_for_action_thread = threading.Thread(target=self.ask_for_action, args=(question,)) # 初始化线程
                    self.ask_for_action.start() # 启动线程

                print(f"\n大模型进行任务分类: ", end='')
                task = get_task_type(question) # 任务分类
                print(task)
                
                if STATUS.TAKE_ACTION:
                    # 等待识别动作的大模型返回结果
                    if STATUS.TAKE_ACTION:
                        self.ask_for_action_thread.join()
                    if '没有动作' in STATUS.Action_from_User:
                        pass
                    else:
                        task = 'qa'
            
            # 在讲解过程中被打断，并且不准备继续之前的讲解，就可以清除相关变量
            if 'ExplainClass' in name and 'continue' not in task:
                self.sentences = [] # 清空句子列表
                self.index_of_sentence = 0 # 重置句子索引

            if 'visit' in task:
                target_point = task.split(" ")[1] # 从字符串中提取目标点（如'卫星展厅'、'下一个'）

                # TODO: 思考不同任务过程中的新导航指令是否需要不同的处理（可能与更改目标点列表策略有关）

                if '下一个' in target_point: # 去往下一个展厅

                    if 'NavigationClass' in name: # 如果是导航中被打断
                        now_destination = STATUS.Current_Order_of_Visit[0]

                        if len(STATUS.Current_Order_of_Visit) > 1: # 下一个展厅是目标点列表中的第二个展厅
                            text2speech(f"跳过{now_destination}，去往下一个展厅", index=1000)
                            STATUS.set_Destination_Area(new_Destination_Area=STATUS.Current_Order_of_Visit[1])

                        else: # 如果已经没有下一个展厅
                            text2speech(f"不好意思，{now_destination}已经是最后一个展厅，请您告诉我，想去哪个展厅还是继续去{now_destination}", index=1000)
                            pn_flag = True
                            continue
                    
                    else: # 如果并非导航过程中被打断，说明上一次导航已经结束

                        if len(STATUS.Current_Order_of_Visit) > 0: # 下一个展厅是目标点列表中的第一个展厅
                            text2speech("去往下一个展厅", index=1000)
                            STATUS.set_Destination_Area(new_Destination_Area=STATUS.Current_Order_of_Visit[0])
                        
                        else: # 如果已经没有下一个展厅
                            text2speech("不好意思，已经没有下一个展厅，请您向我提问或告诉我您要去哪个展厅", index=1000)
                            pn_flag = True
                            continue
                
                elif '上一个' in target_point:  # 去往上一个展厅

                    if STATUS.Last_Area != None: # 如果有上一个展厅的话
                        text2speech("去往上一个展厅", index=1000)
                        STATUS.set_Destination_Area(new_Destination_Area=STATUS.Last_Area)

                    else: # 如果没有上一个展厅
                        text2speech("不好意思，我们已经在第一个展厅，请您向我提问或告诉我您要去哪个展厅", index=1000)
                        pn_flag = True
                        continue
                    
                else: # 不是'下一个'或'上一个'，而是直接给出目标点名称
                    STATUS.set_Destination_Area(new_Destination_Area=target_point)

            elif 'sleep' in task:
                pass
            
            elif 'continue' in task:
                return 'continue'

            else:
                pass

        # 如果有明确的visit指令，或者导航过程被打断后的continue指令，那就打印将要去的目标点
        if 'visit' in task or ('NavigationClass' in name and 'continue' in task):
            print("\n即将去往: ", STATUS.Destination_Area)
            print("\n目前的目标点列表: ", STATUS.Current_Order_of_Visit)

        STATUS.set_is_QAing(False) # 防止问答被打断 - 结束

    def ask_for_action(self, question):
        print("\n开始识别用户希望机器人执行的动作")
        STATUS.Action_from_User = None
        STATUS.Action_from_User = ask(question)
        if not STATUS.Action_from_User == '没有动作':
            # TODO: 向下位机发送动作指令
            print(f"\n向下位机发送 {STATUS.Action_from_User} 动作指令")
        else:
            print("\n用户不希望机器人执行动作")
    
    def get_self_name(self, ):
        return self.__class__.__name__


class QAClass():
    def __init__(self):
        self.qaclass_thread = threading.Thread(target=self.QAClassInit) # 初始化问答模型的线程
        self.qaclass_thread.start() # 启动问答模型的初始化
    
    def QAClassInit(self):
        QaClass = get_llm_answer(model='huozi')
        self.qaclass = QaClass()
    
    def answer_question(self, user_words:str) -> str:
        STATUS.set_is_QAing(True)

        text2speech("请稍等，让我思索一下。", index=0)
        
        # 文档检索问答模型初始化完成
        self.qaclass_thread.join()

        qa_answer = self.qaclass.process_query(user_words) # 问答模型处理用户问题
        text2speech(qa_answer, index=1000) # 语音回答用户问题
        text2speech('我说完了，大家还有问题吗？', index=1000)

        STATUS.set_is_QAing(False)
        

class NavigationClass(InterruptClass):
    def __init__(self, qa_class:object):
        super().__init__(qa_class) # 调用父类的初始化方法

        global STATUS
        
        # 可能需要的变量
        self.stopped = False # 机器人是否停下
        self.arrived = False # 机器人是否已经到达
        self.error_code = None # 记录异常情况类别
        
    def request_service_and_send_destination(self, destination: int):
        """请求导航服务并发送目标点代号。

        Args:
            destination_code (int): 目标点的代号。
        """
        print(f"\n发送去往 {destination} 号目标点的信号")
        self.stopped  = False
        self.arrived = False
        
        # try:
        #     leg_service = rospy.wait_for_service("command_msg") # 等待与下位机通信的服务变得可用
        #     command_client = rospy.ServiceProxy("command_msg", msgCommand) # 创建服务代理，用于调用服务
        #     response = command_client(destination) # 向服务发送请求，阻塞，直至到达目标点才会返回
        # except rospy.ROSException as e: # 如果通信异常
        #     rospy.loginfo("Timeout waiting for message.")
        #     self.error_code = 'NAVIGATION_SERVICE_CONNECTION_FAILED' # 记录异常
        # self.stopped = True

        print(f"开始行走...")
        i = 0
        while i < 1000000:
            print(f"iiiiiiii: ", i)
            sleep(0.1)
            i += 1
            if STATUS.Block_Navigation_Thread:
                break
        print(f"机器人停止...{i}")
        
        self.stopped = True # 机器人停止
        if not STATUS.Block_Navigation_Thread and not STATUS.is_Interrupted:
            self.arrived = True

    def interrupt_navigation(self):
        """处理用户打断导航的操作。

        """
        STATUS.is_Navigating = True
        STATUS.is_Depth_Obstacle = False
        STATUS.is_Yolo_Obstacle = False

        while True:
            interrupt_flag = self.listen_for_INT() # 监听打断信号
            
            # 打断后做出的处理
            if interrupt_flag:
                # 向下位机发送停止信号
                pub_string = actionStopper()
                pub_string.stopper = True
                STATUS.Stop_Publisher.publish(pub_string)
                print("\n因为语音打断，向下位机发送停止信号")

                if_continue = self.handle_interrupt()  # 调用父类InterruptClass中的方法处理打断
                if not if_continue: # 不是继续就直接结束导航
                    STATUS.is_Navigating = False
                    return False
                else: # 是继续就继续导航
                    pass

            # 如果成功到达指定地点
            if self.is_navigation_successful():
                STATUS.is_Navigating = False
                self.arrived = False
                return True
            
            if STATUS.OBSTAC_STOP:
                # 如果遇到障碍物
                if STATUS.is_Depth_Obstacle or STATUS.is_Yolo_Obstacle:
                    # 向下位机发送停止信号
                    STATUS.Block_Navigation_Thread = True
                    pub_string = actionStopper()
                    pub_string.stopper = True
                    STATUS.Stop_Publisher.publish(pub_string)
                    print("\n因为遇到障碍物，向下位机发送停止信号")
                
                    i = 0
                    while STATUS.is_Depth_Obstacle or STATUS.is_Yolo_Obstacle:
                        time.sleep(0.1)
                        if i % 30 == 0:
                            print("\n语音提示用户避让")
                            text2speech("你好，请让一让", index=0)
                        i += 1
                    
                    STATUS.Block_Navigation_Thread = False

                    thread.start_new_thread(self.request_service_and_send_destination, (STATUS.get_first_Current_Order_of_Visit_id(),)) # 启动线程用于向下位机发送导航目标点
            
    def is_navigation_successful(self, ) -> bool:
        """判断是否到达指定地点。

        Returns:
            bool: 如果导航成功完成，到达指定地点，返回True，否则返回False。
        """
        if self.arrived and self.error_code != 'NAVIGATION_SERVICE_CONNECTION_FAILED':
            return True


class ExplainClass(InterruptClass):
    def __init__(self, qa_class:object):
        super().__init__(qa_class)

        global STATUS
        
        # 可能需要的变量
        self.interrupted = False # 是否被用户语音打断
        self.explanation_completed = False # 讲解是否完整完成
        self.error_code = None # TODO: 讲解过程中，喇叭的错误处理

        self.sentences = [] # 用于存储拆分后的所有句子
        self.index_of_sentence = 0 # 记录当前讲解到的句子的索引

    def get_config_explanation_words(self, config_path, hall_index):
        """从配置文件中加载讲解内容。

        Args:
            config_path (str): 配置文件的路径。
            hall_index (int): 展厅的代号。
        """
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        all_speech = config['讲解词列表']
        text = all_speech[STATUS.Origin_Order_of_Visit[hall_index]]

        return text

    def split_and_speech_text(self, hall_index):
        """读取文本并按句子拆分。

        Args:
            hall_index (int): 展厅的代号。

        Returns:
            bool: 是否讲解成功。
        """
        STATUS.is_Explaining = True

        self.sentences = []

        # 从配置文件中加载讲解内容
        config_path = '/home/kuavo/catkin_dt/src/voice_pkg/scripts/config/commentary_config.yaml'
        text = self.get_config_explanation_words(config_path, hall_index) # 从配置文件中加载讲解内容

        splitters = [',', ';', '.','!','?',':', '，', '。', '！', "'", '；', '？', '：', '/'] # 遇到这些符号就拆分句子
        numbers = [str(c) for c in range(10)] # 遇到数字不拆分句子

        sentence = "" # 用于存储句子
        pre_c = '' # 用于存储上一个字符

        # 遍历文本，切分段落为句子
        for c in (text):
            sentence += c

            # 一旦碰到标点符号，就判断目前的长度
            if c in splitters and len(sentence) > 8:
                if pre_c in numbers:
                    continue
                self.sentences.append(sentence) # 加入句子列表
                sentence = ""
        if sentence:
            self.sentences.append(sentence)

        # 依次播放句子
        while self.index_of_sentence < len(self.sentences):
            sentence_index = 0
            for sentence_index in range(self.index_of_sentence, len(self.sentences)):
                self.index_of_sentence += 1 # 现在讲到的句子的索引加一

                sentence_for_read = self.sentences[sentence_index]

                # 发送文本到喇叭
                print("发送文本到喇叭:", sentence_for_read)

                text2speech(text=sentence_for_read, index=0)  # 0表示异步播放
                
                # 处理打断
                interrupt_flag = self.listen_for_INT() # 监听打断信号
                if interrupt_flag:  # 打断后做出的处理
                    if_continue = self.handle_interrupt()  # 调用父类InterruptClass中的方法处理打断
                    if not if_continue: # 不是继续就直接结束讲解
                        STATUS.is_Explaining = False
                        return False
                    else: # 是继续就从上一句没说完的话开始讲
                        self.index_of_sentence -= 1
                        break
                
                # 播放到最后一个句子
                if sentence_index == len(self.sentences) - 1:
                    self.explanation_completed = True # 讲解完成
                    self.sentences = [] # 清空句子列表
                    self.index_of_sentence = 0 # 重置句子索引
                    break
        
        if self.explanation_completed:
            STATUS.is_Explaining = False
            return True
                
    def is_narration_successful(self):
        """判断是否讲解成功。

        Returns:
            bool: 如果讲解成功完成，返回True，否则返回False。
        """ 

        if self.explanation_completed and self.error_code != 'TTS_ERROR':
            return True
        

class MainClass:
    def __init__(self):
        # 实例化所有任务的类
        self.qa_class = QAClass()
        self.start = InterruptClass(self.qa_class)
        self.navigation_class = NavigationClass(self.qa_class)
        self.explain_class = ExplainClass(self.qa_class)

        self.temp_last_area = None # 上一个目标点只有在下一次导航开启的时候才会被更新

        # 设置所有功能开关
        STATUS.set_TAKE_ACTION(False)
        STATUS.set_FACE_DETECT(False)
        STATUS.set_OBSTAC_STOP(True)

        rospy.init_node('interrupt', anonymous=True)
        self.ivw_sub = rospy.Subscriber("ivw_chatter", String, self.ivw_callback) # 订阅ivw话题，唤醒词
        if STATUS.FACE_DETECT:
            self.face_sub = rospy.Subscriber("face_chatter", String, self.face_callback)  # 订阅face话题， 检测人脸
        if STATUS.OBSTAC_STOP:
            self.obs_sub = rospy.Subscriber("obs_chatter", String, self.obs_callback)  # 订阅obs话题，停障
            self.yolo_sub = rospy.Subscriber("yolo_chatter", String, self.yolo_callback)  # 订阅yolo话题，停障
        if STATUS.POSE_DETECT:
            self.pose_sub = rospy.Subscriber("pose_chatter", String, self.pose_callback)
        
        STATUS.set_Stop_Publisher(rospy.Publisher("stopper_msg", actionStopper, queue_size=1))

        self.welcome()

    def ivw_callback(self, data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s from ivw", data.data)
        # 打断信号置True
        if not STATUS.is_QAing:
            print(f"打断回调函数接收到打断信号，可以打断")
            STATUS.set_is_Interrupted(True)
            STATUS.set_Block_Navigation_Thread(True)
        else:
            print(f"打断回调函数接收到打断信号，不可以打断")

    def face_callback(self, data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s from face", data.data)
        print(f"人脸回调函数接收到人脸信号")
        if data.data == "NOBIGFACE":
            STATUS.set_Big_Face_Area("NOBIGFACE")
        elif data.data == "LEFT":
            STATUS.set_Big_Face_Area("LEFT")
        elif data.data == "RIGHT":
            STATUS.set_Big_Face_Area("RIGHT")
        else:
            STATUS.set_Big_Face_Area("CENTER")
            
        if STATUS.Big_Face_Area == "CENTER":
            STATUS.set_is_Big_Face_Detected(True) # 检测到居中人脸
        else:
            STATUS.set_is_Big_Face_Detected(False) # 没有检测到居中人脸，需要转向
        
    def obs_callback(self, data):
        if data.data == '++++':
            if STATUS.is_Depth_Obstacle == False:
                print(f"深度回调函数接收到障碍物信号")
            STATUS.set_is_Depth_Obstacle(True)
        else:
            STATUS.set_is_Depth_Obstacle(False)

    def yolo_callback(self, data):
        if data.data == '++++':
            if STATUS.is_Yolo_Obstacle == False:
                print(f"YOLO回调函数接收到障碍物信号")
            STATUS.set_is_Yolo_Obstacle(True)
        else:
            STATUS.set_is_Yolo_Obstacle(False)

    def pose_callback(self, data):
        pass

    # TODO: 和播音并行，开一个线程使用service串行执行随机的动作
    # TODO: 让GPT4对文本进行切分并匹配动作（动作的执行时间应该与文本的播音时间匹配）尤其是说到数字时伸手指
    # TODO: 加上手势识别
    
    def welcome(self, ):
        STATUS.set_is_QAing(True)
        # text2speech("如果需要我，请说，“夸父同学”", index=1000)
        time.sleep(0.5)
        STATUS.set_is_QAing(False)
        
        # 麦克风不可用，手动唤醒
        msg = String('夸父同学')
        self.ivw_callback(msg)
        
        while True:
            if STATUS.is_Interrupted:
                STATUS.is_Interrupted = False
                STATUS.Block_Navigation_Thread = False
                self.main()
                break
        
    def main(self, ):
        while True:
            print(f"目标点列表: {STATUS.Current_Order_of_Visit}")

            next_destination = STATUS.get_first_Current_Order_of_Visit_id() # 获取下一个目标点
            
            if next_destination != None: # 如果还有目标点，执行导航
                print(f"\n执行去往 {next_destination} 号目标点的导航")

                text2speech(f"下面带大家参观{STATUS.Current_Order_of_Visit[0]}", index=1000)
                thread.start_new_thread(self.navigation_class.request_service_and_send_destination, (next_destination,)) # 启动线程用于向下位机发送导航目标点

                STATUS.Last_Area = self.temp_last_area # 下一次导航开始时，记录上一次导航的目标点
                if_success_navigate = self.navigation_class.interrupt_navigation() # 处理导航过程中的打断

                print("\nif_success_navigate: ", if_success_navigate)
                if if_success_navigate: # 如果导航成功，执行讲解
                    # 修改当前位置
                    self.temp_last_area = STATUS.Current_Order_of_Visit.pop(0)
                    STATUS.Current_Area = STATUS.Origin_Order_of_Visit[next_destination]

                    print(f"\n执行位于 {next_destination} 号目标点的讲解")

                    if_success_explain = self.explain_class.split_and_speech_text(next_destination) # 启动讲解

                    print("\nif_success_explain: ", if_success_explain)
                    if if_success_explain: # 如果讲解成功，执行问答
                        print(f"\n执行位于 {next_destination} 号目标点的问答")

                        self.start.handle_interrupt()
                
                # 重置flag值
                if_success_navigate = False
                if_success_explain = False

            else: # 如果没有目标点了，全流程结束
                break

        text2speech("参观到此结束，欢迎再来哈工大找我玩。", index=1000)
        print("任务结束")
    
if __name__ == "__main__":
    main_class = MainClass()