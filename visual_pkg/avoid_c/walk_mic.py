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

from std_msgs.msg import *

# sys.path.append(rospkg.RosPack().get_path('leju_lib_pkg'))
sys.path.append('/home/lemon/robot_ros_application/catkin_ws/src/leju_lib_pkg')
sys.path.append('/home/lemon/catkin_zt/src/zt_ros/scripts')
sys.path.append('/home/lemon/catkin_zt/src')
import motion.bodyhub_client as bodycli
from speech import StreamPlay, speech_detect

STEP_LEN = [0.1, 0.05, 10.0]


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
        print('\n%-15s%s' % (' ', '--forward'))
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

    def walking_n_steps(self, x, y, a, n):
        for i in range(0, n):
            self.walking(x, y, a)
            print('%s %-10s%-10s%-10s%-10s' % ('step', i+1, x, y, a))
        self.printTeleInfo()

    def getch(self, str=''):
        print(str),
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print(ch)
        return ch

    def getKey(self, key_timeout):
        settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], key_timeout)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ' '
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key
    
    def getKey_mic(self,):
        instruct = speech_detect()
        print(instruct)
        if '前进' in instruct:
            return 'w'
        if '后退' in instruct:
            return 's'
        if '结束' in instruct:
            return 'q'


    def keyboard_thread(self, args):
        while not rospy.is_shutdown():
            self.key_val = self.getKey_mic()
            # self.key_val = self.getKey(0.5)
            if self.key_val == 'q':
                rospy.signal_shutdown('exit')
                return

    def control_thread(self, args):
        w_cmd = [0.0, 0.0, 0.0]
        while not rospy.is_shutdown():
            self.walking_wait()
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
            self.key_val = ' '

    def start(self):
        self.bodyhub_walk()
        self.printTeleInfo()
        try:
            _thread.start_new_thread(self.keyboard_thread, (None,))
            _thread.start_new_thread(self.control_thread, (None,))
            while not rospy.is_shutdown():
                pass
        except:
            print("Error: unable to start _thread")
        rospy.signal_shutdown('exit')


if __name__ == '__main__':
    KeyTele().start()
