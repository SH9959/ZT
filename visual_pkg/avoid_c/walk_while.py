# -*- coding: utf-8 -*-

import sys
import math
import time
import tty
import termios
import select
import _thread

import rospy
import rospkg
import pyrealsense2 as rs
import numpy as np
from std_msgs.msg import *
import sys

# 移除 ros 中 Python 路径
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
# 导入包过后重新添加回 ros 中的 Python 路径
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import motion.bodyhub_client as bodycli

STEP_LEN = [0.1, 0.05, 10.0]

def detect_obstacle():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)
    try:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()
        # if not depth_frame or not color_frame:
            # continue
        # Convert images to numpy arrays

        depth_image = np.asanyarray(depth_frame.get_data())

        # color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # remove bg
        depth_back = depth_image.copy()
        depth_back = np.where(depth_back < 1000, depth_back, 0)
        if np.sum(depth_back) == 0:
            return None

        meidan = np.median(depth_back[(depth_back < 1000) & (depth_back > 0)])
        mask = depth_back.copy()
        mask[mask > meidan + 700] = 0
        mask[mask < meidan - 700] = 0
        mask[mask > 0] = 1

        mask = mask.astype('uint8')
        # _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(mask, 2, 1)
        contours = sorted(contours, key=cv2.contourArea)
        out_mask = np.zeros_like(depth_back)
        # 选面积最大的
        cnt = contours[-1]
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
        # depth_back_colormap = cv2.applyColorMap(cv2.convertScaleAbs(out_mask, alpha=0.03), cv2.COLORMAP_JET)
        # cv2.putText(depth_back_colormap, f"Shape center: ({shape_middle_x},{shape_middle_y})", (depth_back.shape[0]//2, depth_back.shape[1]//2), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(0, 255, 0))
        return shape_middle_x, shape_middle_y
    except:
        # Stop streaming
        pipeline.stop()
        return None
    finally:
        pipeline.stop()


class Action(object):
    '''
    robot action
    '''

    def __init__(self, name, ctl_id):
        rospy.init_node(name, anonymous=True)
        time.sleep(0.2)
        rospy.on_shutdown(self.__ros_shutdown_hook)

        self.bodyhub = bodycli.BodyhubClient(ctl_id)

    def __ros_shutdown_hook(self):
        if self.bodyhub.reset() == True:
            rospy.loginfo('bodyhub reset, exit')
        else:
            rospy.loginfo('exit')

    def bodyhub_ready(self):
        if self.bodyhub.ready() == False:
            rospy.logerr('bodyhub to ready failed!')
            rospy.signal_shutdown('error')
            time.sleep(1)
            exit(1)

    def bodyhub_walk(self):
        if self.bodyhub.walk() == False:
            rospy.logerr('bodyhub to walk failed!')
            rospy.signal_shutdown('error')
            time.sleep(1)
            exit(1)

    def start(self):
        print('action start')


class KeyTele(Action):
    def __init__(self):
        super(KeyTele, self).__init__('key_telecontrol', 2)

        self.key_val = ' '
        self.step_len = STEP_LEN
        self.timeout = 15
        self.gait_cmd_pub = rospy.Publisher('/gaitCommand', Float64MultiArray, queue_size=2)

    def printTeleInfo(self):
        print('\n%-15s%s' % (' ', 'w--forward'))
        print('%-15s%-15s%-15s' % ('a--left', 's--backward', 'd--right'))
        print('%-15s%-15s%-15s' % ('z--trun left', 'x--in situ', 'c--trun right'))
        print('%-15s%s\n' % (' ', 'q--quit'))

    def walking_wait(self):
        rospy.wait_for_message('/requestGaitCommand', Bool, self.timeout)

    def walking_send(self, delta):
        self.gait_cmd_pub.publish(data=delta)

    def walking(self, delta_x, delta_y, theta):
        rospy.wait_for_message('/requestGaitCommand', Bool, self.timeout)
        self.gait_cmd_pub.publish(data=[delta_x, delta_y, theta])

    def realsense_thread(self, args):
        while not rospy.is_shutdown():
            res = detect_obstacle()
            print('hihi: ', res==None)
            if res == None:
                key = ' '
            else:
                shape_middle_x, shape_middle_y = res
                if shape_middle_x > 640 // 2 and shape_middle_x < 640 - 100:
                    key = 'z'
                elif shape_middle_x < 640 // 2 and shape_middle_x > 100:
                    key = 'c'
                else:
                    key = ' '
            self.key_val = key
        
    def control_thread(self, args):
        w_cmd = [0.0, 0.0, 0.0]
        while not rospy.is_shutdown():
            self.walking_wait()
            # print('a')
            if self.key_val == 'w':
                w_cmd = [self.step_len[0], 0.0, 0.0]
            elif self.key_val == 's':
                w_cmd = [-self.step_len[0]*0.8, 0.0, 0.0]
            elif self.key_val == 'a':
                w_cmd = [0, self.step_len[1], 0.0]
            elif self.key_val == 'd':
                w_cmd = [0, -self.step_len[1], 0.0]
            elif self.key_val == 'z':
                w_cmd = [0.0, 0.0, self.step_len[2]]
            elif self.key_val == 'c':
                w_cmd = [0.0, 0.0, -self.step_len[2]]
            elif self.key_val == 'x':
                w_cmd = [0.0, 0.0, 0.0]
            elif self.key_val == ' ':
                continue
            self.walking_send(w_cmd)

    def start(self):
        self.bodyhub_walk()
        self.printTeleInfo()
        try:
            _thread.start_new_thread(self.realsense_thread, (None,))
            _thread.start_new_thread(self.control_thread, (None,))
            while not rospy.is_shutdown():
                pass
        except:
            print("Error: unable to start _thread")
        rospy.signal_shutdown('exit')


if __name__ == '__main__':
    KeyTele().start()
