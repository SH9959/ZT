import rospy
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
from std_msgs.msg import String

import cv2
import numpy as np # 矩阵运行库
# import message_filters # 同步双串口输出
import onnxruntime as ort  # 模型推理库
import time
import argparse  # 参数设置库

# ---parameters---
fx, fy, ppx, ppy = 604.3232421875, 603.890808105469, \
    327.004364013672, 239.296356201172 # 相机内参
horiBound = [338, 470]
vertiBound = [120, 590]

# YOLO 5
class yolov5():
    """
        构造Yolov5运行类
    """

    def __init__(self, modelpath, confThreshold=0.5, nmsThreshold=0.5, objThreshold=0.5):
        with open('/home/kuavo/catkin_dt/src/checkpoints/yolov5weight/class.names', 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')  # 类别列表
        self.num_classes = len(self.classes)  # 类别个数
        if modelpath.endswith('6.onnx'):
            self.inpHeight, self.inpWidth = 1280, 1280
            anchors = [[19, 27, 44, 40, 38, 94], [96, 68, 86, 152, 180, 137], [140, 301, 303, 264, 238, 542],
                       [436, 615, 739, 380, 925, 792]]
            self.stride = np.array([8., 16., 32., 64.])
        else:
            self.inpHeight, self.inpWidth = 640, 640
            anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
            self.stride = np.array([8., 16., 32.])
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [np.zeros(1)] * self.nl
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession(modelpath, so)
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold
        # self.inpHeight, self.inpWidth = (self.net.get_inputs()[0].shape[2], self.net.get_inputs()[0].shape[3])

    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.inpWidth, self.inpHeight
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.inpWidth - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.inpWidth - neww - left, cv2.BORDER_CONSTANT,
                                         value=(114, 114, 114))  # add border
            else:
                newh, neww = int(self.inpHeight * hw_scale), self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.inpHeight - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.inpHeight - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(114, 114, 114))
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img

    def postprocess(self, frame, outs, padsize=None):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        newh, neww, padh, padw = padsize
        ratioh, ratiow = frameHeight / newh, frameWidth / neww
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.

        confidences = []
        boxes = []
        classIds = []
        for detection in outs:
            if detection[4] > self.objThreshold:
                scores = detection[5:]
                classId = np.argmax(scores)
                if classId != 0:
                    continue
                confidence = scores[classId] * detection[4]
                if confidence > self.confThreshold:
                    center_x = int((detection[0] - padw) * ratiow)
                    center_y = int((detection[1] - padh) * ratioh)
                    width = int(detection[2] * ratiow)
                    height = int(detection[3] * ratioh)
                    left = int(center_x - width * 0.5)
                    top = int(center_y - height * 0.5)

                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
                    classIds.append(classId)
        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        # indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold).flatten()
        # indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        indices = np.array(cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)).flatten()
        bottom_line_lowest = 0
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
            if left < 80 or (left + width) > 560:
                continue
            bottom_line_lowest = max(bottom_line_lowest, top + height)
        return frame, bottom_line_lowest

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        # if classId == 0:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return frame

    def detect(self, srcimg):
        img, newh, neww, padh, padw = self.resize_image(srcimg)
        img = self.preprocess(img)
        # Sets the input to the network
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0].squeeze(axis=0)

        # inference output
        row_ind = 0
        for i in range(self.nl):
            h, w = int(img.shape[0] / self.stride[i]), int(img.shape[1] / self.stride[i])
            length = int(self.na * h * w)
            if self.grid[i].shape[2:4] != (h, w):
                self.grid[i] = self._make_grid(w, h)

            outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
                self.grid[i], (self.na, 1))) * int(self.stride[i])
            outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
                self.anchor_grid[i], h * w, axis=0)
            row_ind += length
        srcimg, bottom_line = self.postprocess(srcimg, outs, padsize=(newh, neww, padh, padw))
        return srcimg, bottom_line

class YoloPub():
    def __init__(self) -> None:
        self.color_frame_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.color_callback)
        self.depth_frame_sub = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)
        self.color = np.empty([480,640,3], dtype = np.uint8)
        self.depth = np.empty([480,640], dtype = np.float64)
        self.bridge = CvBridge()
        # 设置 yolov5 网络
        modelpath = '/home/kuavo/catkin_dt/src/checkpoints/yolov5weight/yolov5s.onnx'
        confThreshold = 0.3
        nmsThreshold = 0.5
        objThreshold = 0.3
        self.yolonet = yolov5(modelpath, confThreshold=confThreshold, nmsThreshold=nmsThreshold, objThreshold=objThreshold)
        
        # 运行网络
        self.yolo_detect()

    def color_callback(self, data):  
        self.color = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def depth_callback(self, data):
        self.depth = self.bridge.imgmsg_to_cv2(data, "16UC1")

    def yolo_detect(self):
        cv2.namedWindow('yolo', cv2.WINDOW_AUTOSIZE)
        while not self.depth.any() or not self.color.any():
            # 如果都是空的，也就是一张图片都还没来，就循环等待相机开启
            print('waiting')
            if rospy.is_shutdown():
              break
        pub = rospy.Publisher('yolo_chatter', String, queue_size=1)
        while 1:
            detect_image, bottom_line = self.yolonet.detect(self.color)

            if bottom_line > 300:
              obs_str = "++++"
              rospy.loginfo(obs_str + str(rospy.get_time()))
              pub.publish(obs_str)
            else:
              obs_str = "----"
              rospy.loginfo(obs_str + str(rospy.get_time()))
              pub.publish(obs_str)
            cv2.imshow("yolov5 detect img", detect_image)
            
            key = cv2.waitKey(1)
            if key == 27:
              break
            
            if rospy.is_shutdown():
              break
    
def publisher_node():
    rospy.init_node('I', anonymous=True)
    handler = YoloPub()
    while (not rospy.is_shutdown()):
        pass
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    publisher_node()