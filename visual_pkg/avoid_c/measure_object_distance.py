# ref: https://pysource.com
import rospy
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
from std_msgs.msg import String
import math
# from mask_rcnn import *
# mrcnn = MaskRCNN()

class RealsenseAvoid():
    def __init__(self) -> None:
        self.color_frame_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.color_callback)
        self.depth_frame_sub = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)
        self.color = np.empty([480,640,3], dtype = np.uint8)
        self.depth = np.empty([480,640], dtype = np.float64)
        self.depth_threshold = 800
        self.minx = 11111110
        self.bridge = CvBridge()
        self.process_image()

    def color_callback(self, data):  
        self.color = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def depth_callback(self, data):
        self.depth = self.bridge.imgmsg_to_cv2(data, "16UC1")

        
    def detect_obstacle(self, depth_image):
        ''' angle down!!!! '''
        depth_frame = depth_image
        depth_frame[0:360,:] = 0
        depth_frame[:,:80] = 0
        depth_frame[:,560:640] = 0
        depthsum = np.sum(depth_image) // (480*120*10)
        print(self.minx)
        self.minx = min(depthsum, self.minx)
        if depthsum < 189:
           return 1
        else:
           return 0
        
        '''angle horizontal'''
        # remove bg
        depth_back = depth_image.copy()
        if np.sum(depth_back) < 100000000:
          print('||||||')
          return 100
        depth_back = np.where(depth_back < self.depth_threshold, depth_back, 0)
        if np.sum(depth_back) == 0:
            return 0
        mask = depth_back.copy()
        # meidan = np.median(depth_back[(depth_back < self.depth_threshold) & (depth_back > 0)])
        # print(meidan)
        # meidan = 900
        mask[mask > self.depth_threshold] = 0
        # mask[mask < meidan - 70] = 0
        mask[mask > 0] = 255
        # mask = depth_back[(depth_back < self.depth_threshold) & (depth_back > 0)]
        # mask[mask > self.depth_threshold] = 0
        # mask[mask < int(self.depth_threshold*0.1)] = 0
        # mask[mask > 0] = 255

        maskcnt = mask.astype('uint8')
        # _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(maskcnt, 2, 1)
        contours = sorted(contours, key=cv2.contourArea)
        # 选面积最大的
        area = 0
        # print(len(contours))
        for cnt in contours:
            area += cv2.contourArea(cnt)  # 计算面积
        print(area)
        if area < 10000:
          return 0
        else:
          return area // 10000
        # 以下是计算面积的中心点，用于鲁班的避障
        out_mask = np.zeros_like(depth_back)
        cv2.drawContours(out_mask, [cnt], -1, 255, cv2.FILLED, 1)

        # 找到最大的contour的中心点
        approx = cv2.approxPolyDP(cnt, 0.009*cv2.arcLength(cnt, True), True)
        x_all = []
        y_all = []
        for i, xy in enumerate(approx.ravel()):
            if (i%2 == 0):
                x_all.append(xy)
            else:
                y_all.append(xy)
        shape_middle_x = sum(x_all) // len(x_all)
        shape_middle_y = sum(y_all) // len(x_all)
        return shape_middle_x, shape_middle_y

    def process_image(self, ):
        cv2.namedWindow('depthobject', cv2.WINDOW_AUTOSIZE)
        while not self.depth.any() or not self.color.any():
            # 如果都是空的，也就是一张图片都还没来，就循环等待相机开启
            print('waiting')
            if rospy.is_shutdown():
              break
        timelength = 300
        time_obs_list = np.zeros(timelength)
        pub = rospy.Publisher('obs_chatter', String, queue_size=1)
        while 1:
            # ret, bgr_frame, depth_frame = self.get_frame_stream()
            bgr_frame, depth_frame = self.color, self.depth
            hasobstacle = self.detect_obstacle(depth_frame)
            time_obs_list[timelength-1] = hasobstacle
            if sum(time_obs_list) > timelength//2:
              obs_str = "++++"
              rospy.loginfo(obs_str + str(rospy.get_time()))
              pub.publish(obs_str)
            else:
              obs_str = "----"
              rospy.loginfo(obs_str + str(rospy.get_time()))
              pub.publish(obs_str)
              if sum(time_obs_list[timelength-2:]) == 0:
                time_obs_list[:] = 0
            time_obs_list[:-1] = time_obs_list[1:]
            time_obs_list[-1] = 0

            # # Get object mask
            # boxes, classes, contours, centers = mrcnn.detect_objects_mask(bgr_frame)

            # # Draw object mask
            # bgr_frame = mrcnn.draw_object_mask(bgr_frame)

            # # Show depth info of the objects
            # mrcnn.draw_object_info(bgr_frame, depth_frame)

            # if self.detect_obstacle():
            #     print('has obstacle~~')
            # else:
            #     print('\n')
            # depth_frame = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.255), cv2.COLORMAP_JET)
            # depth_back_colormap = np.hstack((bgr_frame, depth_frame))
            # # cv2.putText(depth_back_colormap, f"Shape center: ({shape_middle_x},{shape_middle_y})", (depth_back_colormap.shape[0]//2, depth_back_colormap.shape[1]//2), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(0, 255, 0))
            # cv2.putText(depth_back_colormap, f"Shape center: ({shape_middle_x},{shape_middle_y})", (shape_middle_x, shape_middle_y), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(0, 255, 0))
            # # print(depth_frame.mean())
            # # _, depth_frame = cv2.threshold(depth_frame, 2, 1000, cv2.THRESH_BINARY)

            # cv2.imshow("depth frame", depth_back_colormap)
            # # cv2.imshow("depth frame", depth_frame)
            # # cv2.imshow("Bgr frame", bgr_frame)

            key = cv2.waitKey(1)
            if key == 27:
              break
            
            if rospy.is_shutdown():
              break

    
def publisher_node():
    rospy.init_node('object_distance_publisher', anonymous=True)
    handler = RealsenseAvoid()
    print(type(handler), handler)
    while (not rospy.is_shutdown()):
        pass
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    publisher_node()