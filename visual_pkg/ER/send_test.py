
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import numpy as np

try:
  import rospy
  from std_msgs.msg import String, Bool, Int32, Float64, Int8
except ImportError as e:
   print(f"import rospy failed:{e}")

rospy.init_node('face_node', anonymous=True)
pub = rospy.Publisher('face_topic', String, queue_size=10)


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
    print(f"sent successfully {message} to face_topic")
    return True
  except:
     print(f"\033[32msend failed\033[0m")
     
if __name__ == '__main__':
    while 1:
        send_to_topic("Face hello")