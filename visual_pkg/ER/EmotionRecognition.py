# ================
# Author: hsong
# ================
from deepface import DeepFace
import face_recognition

import pyrealsense2 as rs

import os, time, cv2, math
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
from functools import wraps
import numpy as np
from contextlib import redirect_stdout
from io import StringIO


try:
  import rospy
  from std_msgs.msg import String, Bool, Int32, Float64, Int8
except ImportError as e:
   print(f"import rospy failed:{e}")


def timer(func):# 一个函数计时器
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} TIME: {end_time - start_time} s")
        return result
    return wrapper

POLICY = 1  #2表示没收到命令时不运行，1表示没收到命令时不发送，但视觉模块还在独立运行
SEND_MODE = True  # 表示发送模式
RUN2 = True

ROS_MODE = True  # 表示ROS模式运行，为True说明需要节点运行

if not ROS_MODE:
  SEND_MODE = False

rospy.init_node('face_node', anonymous=True)
pub = rospy.Publisher('face_chatter', String, queue_size=10)

def send_to_topic(message:str):
  global pub
  # msg_type = None
  
  # get_msg_type = lambda x: {
  #     str: String,
  #     bool: Bool,
  #     int: Int32
  # }.get(x, None)
  
  # # 获取消息类型
  # msg_type = get_msg_type(type(message))

  try:
    stdmsg = String()
    stdmsg.data = message
    pub.publish(stdmsg)
    return True
  except:
     print(f"\033[32msend failed\033[0m")

'''
def listen():
  rospy.init_node('listener', anonymous=True)
  CONTROLL_SIGN = rospy.Subscriber("chatter", String, callback)
  # spin() simply keeps python from exiting until this node is stopped
  rospy.spin()
  return CONTROLL_SIGN
'''

def callback(data):
  global SEND_MODE
  global RUN2
  global POLICY

  if data.data == "OPEN":
    SEND_MODE = True
    if POLICY == 1:
      RUN2 = True
    elif  POLICY == 2:
       RUN2 = True
    
  elif data.data == "CLOSED":
    SEND_MODE = False
    if POLICY == 1:
       RUN2=True
    elif POLICY == 2:
       RUN2=False
  return (SEND_MODE, RUN2)
     

class ER:  # 本来要做情绪检测，现在先做稳定的人脸识别以及数据库比对。
  def __init__(self, imgs_db:str="FaceDB", img_path:Optional[str]="IMGS_for_TEST/dataset/img1.jpg") -> None:
      
      """
      Params:
        imgs_db: The faces database folder path.
        img_path: a image path for initing the class.(not necessary)

      Return:
        None
      """
      assert isinstance(img_path, (str, np.ndarray))
      self.imgs_db = imgs_db
      self.models = [
            "VGG-Face", 
            "Facenet", 
            "Facenet512", 
            "OpenFace", 
            "DeepFace", 
            "DeepID", 
            "ArcFace", 
            "Dlib", 
            "SFace",
          ]
      self.metrics = [
         "cosine", 
         "euclidean", 
         "euclidean_l2"
         ]
      self.backends = [
            'opencv',      # 假阳，不能识别侧脸（不够敏感，假阴），效果还可以
            'ssd',         #    没挂vpn跑不通
            'dlib',        #    模型还没下
            'mtcnn',       # 存在假阳，标准输出暂时不知道怎么关
            'retinaface',  #    跑不通
            'mediapipe',   # 假阳还挺严重,能识别侧脸，敏感度高
            'yolov8',      #    没跑通
            'yunet',       #    没跑通
            'fastmtcnn',   # 有假阳现象，”两个洞“，可识别侧脸，效果还可以
          ]

      self.model = self.models[0]
      self.metric = self.metrics[0]
      self.backend = self.backends[8]  #TODO:这里人脸识别模型，可选，另外可以换一个库face_recognition,在line 282
      self.confusers = None

      self.img_path = img_path

  @timer
  def confusion_analysis(self, ) -> Tuple[bool, List]:  # 

    """检测图片中的困惑者
    Params:
      self:
    Return:
      存在困惑者则返回(True, [confusers])
      否则返回(False, [])
    
    """

    objs = DeepFace.analyze(
       img_path = self.img_path,
       actions = ['age', 'gender', 'race', 'emotion'],
       detector_backend = self.backend
       )

    print(f"analyze_result:\n")
    print(objs)
    print(type(objs))

    '''
    objs:

    [
        {
          'age': 31, 
          'region': {
              'x': 419, 'y': 301, 'w': 919, 'h': 919, 
              'left_eye': (274, 364), 
              'right_eye': (610, 357)
              }, 
            'face_confidence': 0.89, 
            'gender': {'Woman': 99.99992847442627, 'Man': 7.126117793632147e-05}, 
            'dominant_gender': 'Woman', 
            'race': {
                'asian': 0.2651115804604638, 
                'indian': 0.9680511075586741, 
                'black': 0.039857022977293904, 
                'white': 77.68638597049629, 
                'middle eastern': 13.23567409728015, 
                'latino hispanic': 7.804926204975021
                }, 
            'dominant_race': 'white', 
            'emotion': {
                    'angry': 0.06776468184491585, 
                    'disgust': 2.766232279361974e-05, 
                    'fear': 0.32902764720634503, 
                    'happy': 90.37325920899268, 
                    'sad': 0.6396218118591196, 
                    'surprise': 0.2306688947256741, 
                    'neutral': 8.359627178885173
                    }, 
              'dominant_emotion': 'happy'
          }
      ]
    '''

    candidates = self.confusion_picker(objs)

    if len(candidates) > 0:
      return True, candidates
    
    return False, candidates

  def confusion_picker(self, obj:List[Dict]) -> List[Dict]:  # 
      """挑出我们定义的confusers， （情绪检测）
      :Params:
        obj: 原始检测的所有人

      :Return:
        confusers_list:符合我们条件的confusers

      """
      confusers_list = []
      for item in obj:
        if item['emotion']['angry'] >= 0.7:
          if item['emotion']['fear'] >= 0.2 or item['emotion']['sad'] >= 0.2 or item['emotion']['surprise'] >= 0.2:
            print("Confusion detected")
            confusers_list.append(item)

      self.confusers = confusers_list
        
      return confusers_list
  
  @timer
  def search_in_db(self, img_path: Union[str, np.ndarray], threshold:Optional[float]=None) -> object:  # 返回在DB中的路径
      """在数据库中找人

      :Params:
        img_path: 路径或np格式;不使用self.path为了复用函数
        threshold: 相似度阈值，越小越像

      :Returns:

      """

      dfs = DeepFace.find(img_path = img_path,
      db_path = self.imgs_db, 
      model_name = self.model,
      distance_metric = self.metric,
      detector_backend = self.backend,
      enforce_detection = False,    # 关闭因识别不到人脸的报错，默认False
      align = True,
      threshold = threshold,  # cos值越接近0越好
      silent=True
      )

      print(f"\nSearch_result\n")
      print('='*120)
      print(dfs)
      print('='*120)
      return dfs
          
       
  # def crop_and_find_faces(self, visitors:List[Dict]) -> None:  # 如果一张图有多个人脸，需要先裁剪，分别存储,然后逐个寻找历史.沒必要。search自帶多人尋找

  #   for idx, confuser in enumerate(visitors):
  #     region = confuser['region']
  #     img = Image.open(self.img_path)  # 加载原始图片
  #     x, y, w, h = region['x'], region['y'], region['w'], region['h']

  #     face_img = img.crop((x, y, x + w, y + h))

  #     tmp_save_dir = "./tmp"
  #     if not os.path.exists(tmp_save_dir):
  #       os.mkdir(tmp_save_dir)

  #     sp = os.path.join(tmp_save_dir ),f"confuser_{idx}_face.jpg"
  #     face_img.save(sp)  

  #     r = self.search_in_db(img_path=sp)  # 分别搜索
  #     print("\n")
  #     print(r)

  #     if r:
  #         print(f"Find! It is,", r.split("/")[-2])

  @timer
  def get_faces(self, img_path: Union[str, np.ndarray]) -> List[List[int]]:
    """检测一张图片中的所有人脸

    :Params:
      img_path: str 或 np格式

    :Returns:
      r: boundingboxs of faces in th image
    
    """

    method = 1#1使用deepface， 2使用：face_recognition,这个额处理速度有些慢

    r=[]

    if method == 1:
      objs = DeepFace.extract_faces(
        img_path=img_path, 
        detector_backend = self.backend,  #框架，可选，line 126
        enforce_detection = False,    # 关闭因识别不到人脸的报错，默认False
        align = True,

        )
      # print("\033[32m")
      # print("\nDetected faces:")
      #print(objs)
      for obj in objs:
        r.append([obj['facial_area']['x'], obj['facial_area']['y'], obj['facial_area']['w'], obj['facial_area']['h']])  
      #print(r)
      # print('\033[0m')
      #print("\n")
      if len(r)==0:
        r = []
      else:
        if r[0][0] == 0 and r[0][1] == 0:  # 判定为找不到人脸 [[0,0,640,480]]
          r = []
      print(f'Detected {r}')
    elif method == 2:
      face_locations = face_recognition.face_locations(img_path)
      for face_location in face_locations:        
        x = face_location[3]
        y = face_location[0]
        w = face_location[2] - face_location[0]
        h = face_location[1] - face_location[3]
        r.append([x, y, w, h])
      print(f'Detected {r}')

    return r

  # def update_db(self,) -> None:
      
  #     dfs = DeepFace.find(img_path = self.img_path,
  #     db_path = self.imgs_db, 
  #     model_name = self.model,
  #     distance_metric = self.metric,
  #     detector_backend = self.backend
  #     )
  #     print(f"\nFind_result:")
  #     print(dfs)
  #     return dfs
  
  # def run(self, ):
  #    _, candidate = self.confusion_analysis()
  #    #self.crop_and_find_faces(visitors=candidate)
  #    #self.
  #    pass
  


  
'''
def run_1(img_path:Union[str,np.ndarray]="2.jpg", SEARCH_THRES:Optional[float]=None) -> None:
  """实现判断图像中的人脸，并可视化: IN 或 not IN database，用来单独测试

  :Params:
    img_path: str or np.array
    SEARCH_THRES: 相似度,None的话会自动设置 

  """


  SAVE_PATH = f'{img_path.split(".")[0]}_result.jpg'
  VAR_THRES = 0.8
  MU_THRES = 5
  SEARCH_THRES = SEARCH_THRES
  MAX_WINDOW = 8
  
  RS=False
  CV2=not RS

  NEED_FACE_DETECT = True
      #cv2.imwrite('wmj.jpg',frame)
  tmp = ER()
  frame = cv2.imread(img_path)

  smallFrame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
  # rgbFrame = smallFrame[:, :, ::-1]
  rgbFrame = smallFrame

  faceLandmarks = face_recognition.face_landmarks(rgbFrame)   # 加唇动检测主要是为了人脸规范化过滤，以及加快可视化速度。
  #faceLocations = face_recognition.face_locations(rgbFrame)
  #faceEncodings = face_recognition.face_encodings(rgbFrame, faceLocations)
  Record = []
  for ind, faceLandmark in enumerate(faceLandmarks):
      if len(Record)<ind + 1:
          Record.append([])
          
      p1 = faceLandmark['top_lip']
      p2 = faceLandmark['bottom_lip']
      x1, y1 = p1[9]
      x3, y3 = p1[8]
      x4, y4 = p1[10]
      x2, y2 = p2[9]
      x5, y5 = p2[8]
      x6, y6 = p2[10]
      dist_ = math.sqrt(((x2 + x5 + x6) - (x1 + x3 + x4)) ** 2 + ((y2 + y5 + y6) - (y1 + y3 + y4)) ** 2)
      Record[ind].append(dist_)

  vas = []
  mus = []
  for item in Record:
    if len(item) > MAX_WINDOW:
      item.pop(0)
    va = np.var(item)
    mu = np.mean(item)
    vas.append(va)
    mus.append(mu)
    if va > VAR_THRES  or mu > MU_THRES:  # 最近8帧的方差大于阈值，则说明可能在说话，mu>5表示可能在張嘴，NEED_FACE_DETECT 置为 False
      NEED_FACE_DETECT = False
    

  print("Record:",Record)
  print('vas:', vas)
  print('mus:',mus)


  if not NEED_FACE_DETECT:
     cv2.imwrite(SAVE_PATH, frame)
     print(f"result saved in {SAVE_PATH}")
     return SAVE_PATH

  frame_np = np.array(frame)

  FACE_DETECTED = False

  persons_bounding_boxs = tmp.get_faces(img_path=frame_np)  # 先探测人脸，


  if len(persons_bounding_boxs) > 0 :  # 有人脸
    FACE_DETECTED = True
    # print(f"\033[32mFACE DETECTED\n{persons_bounding_boxs}\033[0m")
    # print('')
    # if len(unkown_persons_bounding_boxs) == 1 and unkown_persons_bounding_boxs[0][0] == 0 and unkown_persons_bounding_boxs[0][1] == 0:  # 全框表示代表没有人脸
    #    print('\033[31mNO FACE DETECED\033[0m')
    # else:
    for id, person in enumerate(persons_bounding_boxs):
      print('\nid',id)
      IN_DB = False
      x, y, w, h = person[0], person[1], person[2], person[3]
      frame_np_tmp = np.array(frame[y:y+h,x:x+w])

      # 裁切人脸区域 单独在database中寻找 分别画框
      result = tmp.search_in_db(img_path=frame_np_tmp, threshold=SEARCH_THRES) 

      if len(result[0]) > 0:
        each_person_matched = result[0]
        IN_DB = True

                # 获取第一个人脸的路径
        image_path = each_person_matched.iloc[0]['identity']  # 获取第一行的 'identity' 列数据
        face_image = cv2.imread(image_path)  # 加载人脸图像
        color = (0, 255, 0)  
        thickness = 1 
        line_type = cv2.LINE_AA 
        # x = each_person_matched.iloc[0]["source_x"] # * scale_rate
        # y = each_person_matched.iloc[0]["source_y"] # * scale_rate
        # w = each_person_matched.iloc[0]["source_w"] # * scale_rate
        # h = each_person_matched.iloc[0]["source_h"] # * scale_rate
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        cv2.rectangle(frame, top_left, bottom_right, color, thickness, lineType=line_type)
        if w >= 3 and h >=3:
          face_image_resized = cv2.resize(face_image,  (w // 3, h // 3))
        else:
          face_image_resized = cv2.resize(face_image,  (3, 3))

        print(f"\033[31m  {x}-{x + int(face_image_resized.shape[0])}  , {y}-{y+int(face_image_resized.shape[1])}  \033[0m")
        frame[y:y + int(face_image_resized.shape[1]), x:x + int(face_image_resized.shape[0])] = face_image_resized

      else:

        color = (0, 0, 255)  #
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (x, y - 10)  # 设置文本的起始位置为人脸框的上方
        fontScale = 0.5
        #color = (0, 0, 255)  # 文本颜色为红色
        thickness = 1
        line_type = cv2.LINE_AA

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness, lineType=line_type)
        cv2.putText(frame, 'UNKNOWN PERSON', org, font, fontScale, color, thickness, line_type)

  else:
      print('\n\033[31mNO FACE DETECED\033[0m\n')
  
  cv2.imwrite(SAVE_PATH,frame)

'''
DEPTH_NEED = False
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
class RealsenseImage():
    def __init__(self) -> None:
        #rospy.init_node('ER_publisher', anonymous=True)
        self.color_frame_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.color_callback)
        if DEPTH_NEED:
          self.depth_frame_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
          self.depth = np.empty([480,640], dtype = np.float64)
        self.color = np.empty([480,640,3], dtype = np.uint8)
        self.bridge = CvBridge()

    def color_callback(self, data):  
        self.color = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def depth_callback(self, data):
        if DEPTH_NEED:
          self.depth = self.bridge.imgmsg_to_cv2(data, "16UC1")
        pass
    
    def getImage(self):
      if DEPTH_NEED:
        while not self.depth.any() or not self.color.any():
          # 如果都是空的，也就是一张图片都还没来，就循环等待相机开启
          # print('waiting')
          if rospy.is_shutdown():
            break
        return self.color, self.depth
      else:
        while  not self.color.any():
          # 如果都是空的，也就是一张图片都还没来，就循环等待相机开启
          # print('waiting')
          if rospy.is_shutdown():
            break

        return self.color

def run_2():  # 流式实时可视化， 和run_1相似
  image_from_ros = RealsenseImage()
  VAR_THRES = 2.8       # 唇动检测的方差阈值
  MU_THRES = 5.0        # 唇动检测的均值阈值
  SEARCH_THRES = 0.35   # 在数据库中查找人脸的相似度阈值，越小越高
  MAX_WINDOW = 8        # 计算唇动检测的窗口大小
  WINDOW_SIZE_AREA = 10 # 人脸面积记录的窗口大小
  AREA_THRES = 0.01     # 人脸面积阈值，小于该值则不发送信号
  SEARCH = False        # 检测到人脸后是否要在数据库中查找？
  global SEND_MODE      # 收到ros主控代码时 发送 信号
  global RUN2           # 为True表示本节点一直运行，False表示只有接收到ROS主控代码时才运行，用前者。

  RS=False              # RealSense相机
  CV2=False            # cv2相机
  ROS=True             # ROS
  LIP_MODE = False      #是否需要检测唇动以跳过人脸检测

  if RS:
    Depth = False
    # pipeline = rs.pipeline()
    # config = rs.config()

    # # Get device product line for setting a supporting resolution
    # pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    # pipeline_profile = config.resolve(pipeline_wrapper)
    # device = pipeline_profile.get_device()
    # device_product_line = str(device.get_info(rs.camera_info.product_line))

    # found_rgb = False
    # for s in device.sensors:
    #     if s.get_info(rs.camera_info.name) == 'RGB Camera':
    #         found_rgb = True
    #         break
    # if not found_rgb:
    #     print("The demo requires Depth camera with Color sensor")
    #     exit(0)
        
    # if Depth:
    #   config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
  
    # Start streaming
    # pipeline.start(config)
    
  if CV2:
    v = 'htg.mp4'
    video_capture = cv2.VideoCapture(v)  
    
  imgs_path_dir = "IMGS_for_TEST/dataset"
  imgs_paths = [os.path.join(imgs_path_dir, f"img{i}.jpg") for i in range(1,68)]

  tmp = ER(img_path=imgs_paths[3])
  # tmp.confusion_analysis()
  # tmp.run()

  Record = []
  Face_Area_record = [0 for _ in range(10)]
  Faces_Area_record = []  # 考虑到一帧可能有多个人脸
  try:
    while True:
      if rospy.is_shutdown():
        break
      try:
        # SEND_MODE, RUN2 = listen()
        pass
      except:
        print("listen failed")
        if not RUN2:
          continue
      

      NEED_FACE_DETECT = True  # 表示需要进行人脸识别，否则直接跳过
      if CV2:
        ret, frame = video_capture.read()
        if not ret:
            break
      if RS:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        if Depth:
          depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
      if ROS:
        # print("DEBUG 31")
        if DEPTH_NEED:
          color_image, depth_image = image_from_ros.getImage()
        else:
          color_image = image_from_ros.getImage() 
        # print("DEBUG 3")
        
        # # Convert images to numpy arrays
        # if Depth:
        #   depth_image = np.asanyarray(depth_frame.get_data())
        # # color_image = np.asanyarray(color_frame.get_data())

        # # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # if Depth:
        #   depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #   depth_colormap_dim = depth_colormap.shape
        # color_colormap_dim = color_image.shape

        # # If depth and color resolutions are different, resize color image to match depth image for display
        # if Depth:
        #   if depth_colormap_dim != color_colormap_dim:
        #       resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        #       images = np.hstack((resized_color_image, depth_colormap))
        #   else:
        #       images = np.hstack((color_image, depth_colormap))
            
        frame = color_image

      # print("DEBUG 4")
      # NEED_FACE_DETECT = True
      # cv2.imwrite('wmj.jpg',frame)
      if LIP_MODE:
        smallFrame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # rgbFrame = smallFrame[:, :, ::-1]
        rgbFrame = smallFrame

        faceLandmarks = face_recognition.face_landmarks(rgbFrame)   # 加唇动检测主要是为了人脸规范化过滤，以及加快可视化速度
        #faceLocations = face_recognition.face_locations(rgbFrame)
        #faceEncodings = face_recognition.face_encodings(rgbFrame, faceLocations)

        for ind, faceLandmark in enumerate(faceLandmarks):
            if len(Record)<ind + 1:
              Record.append([])
              
            p1 = faceLandmark['top_lip']
            p2 = faceLandmark['bottom_lip']
            x1, y1 = p1[9]
            x3, y3 = p1[8]
            x4, y4 = p1[10]
            x2, y2 = p2[9]
            x5, y5 = p2[8]
            x6, y6 = p2[10]
            dist_ = math.sqrt(((x2 + x5 + x6) - (x1 + x3 + x4)) ** 2 + ((y2 + y5 + y6) - (y1 + y3 + y4)) ** 2)
            
            #print(dist)

            Record[ind].append(dist_)
        vas = []
        mus = []
        for item in Record:
          if len(item) > MAX_WINDOW:
            item.pop(0)
          va = np.var(item)
          mu = np.mean(item)
          vas.append(va)
          mus.append(mu)
          if va > VAR_THRES  or mu > MU_THRES:  # 最近8帧的方差大于阈值，则说明可能在说话，mu>5表示可能在張嘴，NEED_FACE_DETECT 置为 False
            NEED_FACE_DETECT = False
          
        print("Record:",Record)
        print('vas:', vas)
        print('mus:',mus)

        if not NEED_FACE_DETECT:
          print("\033[31mFACE NOT NORMALIZED. Please maintain a normal expression.\033[0m")
          cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
          cv2.imshow('RealSense', frame)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
          
          continue
      #scale_rate = 4
      #small_frame = cv2.resize(frame, (frame.shape[0] // scale_rate, frame.shape[1] // scale_rate ))
      #cv2.imwrite("songhao327.jpg", frame)

      frame_np = np.array(frame)
      height, width = frame.shape[:2]
      S = height * width
      #print("DEBUG:1")
      persons_bounding_boxs = tmp.get_faces(img_path=frame_np)
      #print("DEBUG:2")
      if len(persons_bounding_boxs) > 0 :  # 有人脸
        if SEND_MODE:
          faces_num = len(persons_bounding_boxs)
          while len(Faces_Area_record) > faces_num:
            Faces_Area_record.pop(0)
          # 人脸面积在大于阈值时才发
          # 
          for id, person in enumerate(persons_bounding_boxs):
            if len(Faces_Area_record) < id + 1:
              Faces_Area_record.append([])

            x0, y0, w0, h0 = person[0], person[1], person[2], person[3]
            s = w0 * h0
            Faces_Area_record[id].append(s)
            if len(Faces_Area_record[id]) > WINDOW_SIZE_AREA:
              Faces_Area_record[id].pop(0)
            
            
          # 计算平均人脸面积
          detected_face_s = []  # 记录所有大脸的boungdingbox
          for id, item in enumerate(Faces_Area_record):
            print('id',id)
            x1, y1, w1, h1 = persons_bounding_boxs[id][0], persons_bounding_boxs[id][1], persons_bounding_boxs[id][2], persons_bounding_boxs[id][3]
            sm = np.mean(item)
            ra = sm / S
            print("rate of s / S:", ra)
            if ra > AREA_THRES:  # 大于阈值
              detected_face_s.append(persons_bounding_boxs[id])
              
          if len(detected_face_s) > 0:
            print("detected_big_face_s:", detected_face_s)
            xx=[]  
            ww=[]
            for big_face in detected_face_s:
              xx.append(big_face[0])  
              ww.append(big_face[2]) 
            wmax=max(ww)
            wm=sum(ww)/len(ww)
            xm=sum(xx)/len(xx)
            xc = [x + w / 2 for x, w in zip(xx, ww)]

            xcm=sum(xc)/len(xc)
              # 如果人脸偏左，就发送人脸在左边的信号
            if xcm + wm  < width / 2:
              print(f"\033[32mFace on the left\033[0m")#:{persons_bounding_boxs[id]}
              send_to_topic("LEFT")
            elif xcm > width / 2:
              print(f"\033[32mFace on the right\033[0m")#:{persons_bounding_boxs[id]}
              send_to_topic("RIGHT")
              # 发送信号
            else:
              print(f"\033[32mFace on the center.OK.\033[0m")  # 人脸在中间{persons_bounding_boxs[id]}
              send_to_topic(f"CENTER")  # 可以是别的，还没说
          else:
            # no faces
            print(f"\033[32mNo big faces\033[0m")  # 人脸在中间{persons_bounding_boxs[id]}
            send_to_topic(f"NOBIGFACE")    
          
          #send_to_topic("Faces Detected")  # 可以是别的，还没说

        # print(f"\033[32mFACE DETECTED\n{persons_bounding_boxs}\033[0m")
        # print('')
        # if len(unkown_persons_bounding_boxs) == 1 and unkown_persons_bounding_boxs[0][0] == 0 and unkown_persons_bounding_boxs[0][1] == 0:  # 全框表示代表没有人脸
        #    print('\033[31mNO FACE DETECED\033[0m')
        # else:
        for id, person in enumerate(persons_bounding_boxs):
          #print('\nid',id)
          IN_DB = False  # 表示是否在库中
          x, y, w, h = person[0], person[1], person[2], person[3]
          
          frame_np_tmp = np.array(frame[y:y+h,x:x+w])
          if SEARCH:  # 如果 需要寻找
            result = tmp.search_in_db(img_path=frame_np_tmp, threshold=SEARCH_THRES) 

            if len(result[0]) > 0:
              if SEND_MODE:
                send_to_topic("Face IN Database")
              each_person_matched = result[0]
              IN_DB = True

              # 获取第一个人脸的路径
              image_path = each_person_matched.iloc[0]['identity']  # 获取第一行的 'identity' 列数据
              face_image = cv2.imread(image_path) 
              color = (0, 255, 0)   # bgr 绿色
              thickness = 1 
              line_type = cv2.LINE_AA 
              # x = each_person_matched.iloc[0]["source_x"] # * scale_rate
              # y = each_person_matched.iloc[0]["source_y"] # * scale_rate
              # w = each_person_matched.iloc[0]["source_w"] # * scale_rate
              # h = each_person_matched.iloc[0]["source_h"] # * scale_rate
              top_left = (x, y)
              bottom_right = (x + w, y + h)
              
              if w >= 3 and h >=3:
                face_image_resized = cv2.resize(face_image,  (w // 3, h // 3))
              else:
                face_image_resized = cv2.resize(face_image,  (3, 3))
              #print(f"\033[31m  {x}-{x + int(face_image_resized.shape[0])}  , {y}-{y+int(face_image_resized.shape[1])}  \033[0m")
              cv2.rectangle(frame, top_left, bottom_right, color, thickness, lineType=line_type)
              frame[y:y + int(face_image_resized.shape[1]), x:x + int(face_image_resized.shape[0])] = face_image_resized

            else:  # 未在INDB
              # IN_DB = False
              color = (0, 0, 255)  # bgr 红色
              font = cv2.FONT_HERSHEY_SIMPLEX
              org = (x, y - 10)  # 设置文本的起始位置为人脸框的上方
              fontScale = 0.5
              thickness = 1
              line_type = cv2.LINE_AA
              cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness, lineType=line_type)
              cv2.putText(frame, 'UNKNOWN PERSON', org, font, fontScale, color, thickness, line_type)
          else:
            color = (255, 0, 0)  # 蓝色
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (x, y - 10)  # 设置文本的起始位置为人脸框的上方
            fontScale = 0.5
            thickness = 1
            line_type = cv2.LINE_AA
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness, lineType=line_type)
             
      else:
          print('\n\033[31mNO FACE DETECED\033[0m\n')

        #print(result)
      cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
      cv2.imshow('RealSense', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  # 释放视频流

  finally:
    if RS:
    # Stop streaming
      pipeline.stop()
    if CV2:
      video_capture.release()
    cv2.destroyAllWindows()
   
   
if __name__=="__main__":
  # Configure depth and color streams\
  # run_1(SEARCH_THRES=0.3)
  # import argparse
  # import os
 
  #   # parse args
  # parser = argparse.ArgumentParser(description='test')
  # parser.add_argument('-d', '--debug', action='store_true', help='debug mode')
  # args, _ = parser.parse_known_args()
  # if args.debug:
  #     # if you use vscode on hpc-login-01
  #     import debugpy
  #     debugpy.connect(('192.168.50.202', 5901))
  #     debugpy.wait_for_client()
  #     debugpy.breakpoint()

  run_2()
     

'''
# deepface调用示例

#face verification
result = DeepFace.verify(img1_path = imgs_paths[1], 
    img2_path = imgs_paths[2], 
    model_name = models[0],
    distance_metric = metrics[1],
    detector_backend = backends[0]
)

print(f"verify_result:\n")
print(result)

#face recognition
dfs = DeepFace.find(img_path = imgs_paths[1],
    db_path = imgs_db, 
    model_name = models[1],    
#face verification
result = DeepFace.verify(img1_path = imgs_paths[1], 
    img2_path = imgs_paths[2], 
    model_name = models[0],
    distance_metric = metrics[1],
    detector_backend = backends[0]
)

print(f"verify_result:\n")
print(result)

#face recognition
dfs = DeepFace.find(img_path = imgs_paths[1],
    db_path = imgs_db, 
    model_name = models[1],
    distance_metric = metrics[1],
    detector_backend = backends[0]
)
print(f"find_result:\n")
print(dfs)

#embeddings
embedding_objs = DeepFace.represent(img_path = imgs_paths[1], 
    model_name = models[2],
    detector_backend = backends[0]
)

print(f"embedding_result:\n")
print(embedding_objs)

objs = DeepFace.analyze(img_path = imgs_paths[4], 
    actions = ['age', 'gender', 'race', 'emotion'],
    detector_backend = backends[0]
)
print(f"analyze_result:\n")
print(objs)

#face detection and alignment
face_objs = DeepFace.extract_faces(img_path = imgs_paths[4], 
        target_size = (224, 224), 
        detector_backend = backends[4]
)
print(f"extract_faces_result:\n")
print(face_objs)

DeepFace.stream(db_path = imgs_db, model_name=models[0], distance_metric=metrics[0], enable_face_analysis=True)  # 识别到
    distance_metric = metrics[1],
    detector_backend = backends[0]
)
print(f"find_result:\n")
print(dfs)

#embeddings
embedding_objs = DeepFace.represent(img_path = imgs_paths[1], 
    model_name = models[2],
    detector_backend = backends[0]
)

print(f"embedding_result:\n")
print(embedding_objs)

objs = DeepFace.analyze(img_path = imgs_paths[4], 
    actions = ['age', 'gender', 'race', 'emotion'],
    detector_backend = backends[0]
)
print(f"analyze_result:\n")
print(objs)

#face detection and alignment
face_objs = DeepFace.extract_faces(img_path = imgs_paths[4], 
        target_size = (224, 224), 
        detector_backend = backends[4]
)
print(f"extract_faces_result:\n")
print(face_objs)

DeepFace.stream(db_path = imgs_db, model_name=models[0], distance_metric=metrics[0], enable_face_analysis=True)  # 识别到
  
'''