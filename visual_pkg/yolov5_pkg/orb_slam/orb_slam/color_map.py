import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# 标号信息，
# 0：无间测点，继续向前
# 1：观测到障碍物，原地踏步
flag = 0

# 根据描述子信息，筛选特征点
def descriptor(frame, position):
    x, y = position
    blue, green, red = frame[y][x]
    shadow_mask = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_t, s_t, v_t = shadow_mask[y][x]
    distance = 15 # 标准ORB描述子距离为4, 必须小于10
    threshold = 55 # 设置阈值，平均值与中心值相差的绝对值和阈值比较
    blue_sum, green_sum, red_sum = 0, 0, 0
    des_pos = [[x-distance, y], [x-distance, y+1], [x-distance, y-1], [x+distance, y], \
           [x+distance, y+1], [x+distance, y-1], [x+1, y-distance], [x, y-distance], \
            [x-1, y-distance], [x+1, y+distance], [x, y+distance], [x-1, y+distance]]
    for i in range(12):
        blue_temp, green_temp, red_temp = frame[des_pos[i][1]][des_pos[i][0]]
        blue_sum = blue_temp + blue_sum
        green_sum = green_temp + green_sum
        red_sum = red_temp + red_sum
    
    blue_avg, green_avg, red_avg = int(blue_sum/12), int(green_sum/12), int(red_sum/12)

    counter = 0
    if s_t>35:
        if abs(blue_avg-blue) > threshold:
            counter = counter+1
        if abs(green_avg-green) > threshold:
            counter = counter+1
        if abs(red_avg-red) > threshold:
            counter = counter+1
    return counter


class Orb_slam_sub(Node):
    def __init__(self, name):
        super().__init__(name)   
        self.frame = 0     
        self.sub = self.create_subscription(
            Image, "camera/color/image_raw", self.listener_callback, 10
        )
        self.cv_bridge = CvBridge()
        cv2.destroyAllWindows()

    def listener_callback(self, stream_data):
        image = self.cv_bridge.imgmsg_to_cv2(stream_data, "bgr8")
        self.orb(image)
        self.get_logger().info("frame receiving")

    def orb(self, stream_data):
        # shadow mask
        stereo_raw = cv2.cvtColor(stream_data, cv2.COLOR_BGR2GRAY)
        # ORB特征点个数
        orb = cv2.ORB_create(nfeatures=1000)

        # orb提取特征点及描述子
        cur_kps = orb.detect(stereo_raw)
        cur_kps, cur_des = orb.compute(stereo_raw, cur_kps)

        outimg1 = cv2.drawKeypoints(stereo_raw, keypoints=cur_kps, outImage=None, color=(0,0,255))
        
        # Define focus region
        coordinates_raw = cv2.KeyPoint_convert(cur_kps)
        coordinates_raw = np.array(coordinates_raw).astype(dtype=int).tolist()
        coordinates = []
        for i in range(len(coordinates_raw)):
            if (coordinates_raw[i][0]>10) & (coordinates_raw[i][0]<630) \
                & (coordinates_raw[i][1]>10) & (coordinates_raw[i][1]<470):
            # if (coordinates_raw[i][0]>121) & (coordinates_raw[i][0]<590) \
            #     & (coordinates_raw[i][1]>338) & (coordinates_raw[i][0]<470):
                if(descriptor(stream_data, (coordinates_raw[i][0],coordinates_raw[i][1])) >= 2):
                    coordinates.append(coordinates_raw[i])

        coordinates = np.array(coordinates).astype(dtype=int).tolist()
        lower_bound = [0, 0]
        i = 0
        for point in coordinates:
            x, y = point
            if y > lower_bound[1]:
                lower_bound = point
            cv2.circle(stream_data, (point[0], point[1]), 2, (0, 0, 255), 2)
            i += 1

        # for i in range(len(coordinates_raw)):
        #     if (coordinates_raw[i][0]>121) & (coordinates_raw[i][0]<590) \
        #         & (coordinates_raw[i][1]>338) & (coordinates_raw[i][0]<479):
        # 辅助线可视化
        cv2.line(stream_data, (350, 0), (350, 338), (0, 0, 255), 1)
        cv2.line(stream_data, (120, 338), (120, 480), (0, 0, 255), 1)
        cv2.line(stream_data, (590, 338), (590, 480), (0, 0, 255), 1)
        # cv2.line(stream_data, (121, 126), (590, 126), (0, 0, 255), 1) # 2 m 阈值线
        # cv2.line(stream_data, (121, 214), (590, 214), (0, 0, 255), 1) # 1.5 m 阈值线
        cv2.line(stream_data, (121, 338), (590, 338), (0, 0, 255), 1) # 1 m 阈值线

        output = np.hstack([outimg1, stream_data])
        cv2.imshow("final", output)


        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = Orb_slam_sub("orb_slam_sub")

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown