import cv2
import rospy
import numpy as np # 矩阵运行库
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# realsense视频流
class harbinCase:
    def __init__(self):
        self.color_frame_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.color_callback)
        self.depth_frame_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        self.color = np.empty([480,640,3], dtype = np.uint8)
        self.depth = np.empty([480,640], dtype = np.float64)
        self.bridge = CvBridge()
        self.imshow_func()

    def color_callback(self, data):  
        self.color = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def depth_callback(self, data):
        self.depth = self.bridge.imgmsg_to_cv2(data, "16UC1")
    
    def imshow_func(self,):
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        while not self.depth.any() or not self.color.any():
            # 如果都是空的，也就是一张图片都还没来，就循环等待相机开启
            print('waiting')
            continue
        while 1:
            print('showing')
            cv2.imshow('realsensewmj', self.color)
            if(cv2.waitKey(100)==27):
                # 等待100ms，如果用户按下ESC(ASCII码为27),则跳出循环
                break
            if rospy.is_shutdown():
                # 如果终端按下Ctrl+C就跳出循环
                break
def main():
    rospy.init_node('main', anonymous=True)
    handler = harbinCase()
    while (not rospy.is_shutdown()):
        pass
    
if __name__ == '__main__':
    main()