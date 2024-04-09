# ref: https://pysource.com
import rospy
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
from std_msgs.msg import String

from mask_rcnn import *
mrcnn = MaskRCNN()

class RcnnPersonDepth():
    def __init__(self) -> None:
        self.color_frame_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.color_callback)
        self.depth_frame_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        self.color = np.empty([480,640,3], dtype = np.uint8)
        self.depth = np.empty([480,640], dtype = np.float64)
        self.depth_threshold = 800

        self.bridge = CvBridge()
        self.process_image()

    def color_callback(self, data):  
        self.color = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def depth_callback(self, data):
        self.depth = self.bridge.imgmsg_to_cv2(data, "16UC1")

    def process_image(self, ):
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        while not self.depth.any() or not self.color.any():
            # 如果都是空的，也就是一张图片都还没来，就循环等待相机开启
            # print('waiting')
            if rospy.is_shutdown():
              break
        pub = rospy.Publisher('obs_chatter', String, queue_size=1)
        while 1:
            # ret, bgr_frame, depth_frame = self.get_frame_stream()
            bgr_frame, depth_frame = self.color, self.depth
          
            # Get object mask
            boxes, classes, contours, centers = mrcnn.detect_objects_mask(bgr_frame)

            # Draw object mask
            bgr_frame = mrcnn.draw_object_mask(bgr_frame)

            # Show depth info of the objects
            mrcnn.draw_object_info(bgr_frame, depth_frame)


            cv2.imshow("depth frame", depth_frame)
            cv2.imshow("Bgr frame", bgr_frame)

            key = cv2.waitKey(1)
            if key == 27:
              break
            
            if rospy.is_shutdown():
              break

def publisher_node():
    rospy.init_node('rcnn_publisher', anonymous=True)
    handler = RcnnPersonDepth()
    print(type(handler), handler)
    while (not rospy.is_shutdown()):
        pass
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    publisher_node()