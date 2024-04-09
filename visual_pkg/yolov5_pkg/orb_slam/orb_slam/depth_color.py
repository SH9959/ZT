import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np # 矩阵运行库
import message_filters # 同步双串口输出
import argparse  # 参数设置库
import onnxruntime as ort  # 模型推理库
import time

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
        with open('/home/kuavo/Ztest_ws/test/class.names', 'rt') as f:
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
        print(len(boxes))
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame = cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 0, 255), thickness=4)
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
        srcimg = self.postprocess(srcimg, outs, padsize=(newh, neww, padh, padw))
        return srcimg

# 根据深度信息检测，区域范围内是否有小于安全距离的物体
def findOutDistance(frame):
    horiiLine = range(horiBound[0], horiBound[1], 1)
    vertiLine = range(vertiBound[0], vertiBound[1], 1)
    for i in vertiLine:
        for j in horiiLine:
            if (frame[j][i] < 800) and (frame[j][i] != 0):
                print("yes!!!")

            else:
                print("no")

def processing(color_data, depth_data):
    starting_time = time.time()
    # 配置yolov5
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', type=str, default='/home/kuavo/Ztest_ws/src/orb_slam/orb_slam/weights/yolov5s.onnx')
    parser.add_argument('--confThreshold', default=0.3, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    parser.add_argument('--objThreshold', default=0.3, type=float, help='object confidence')
    args = parser.parse_args()
    # 设置网络
    yolonet = yolov5(args.modelpath, confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold,
                     objThreshold=args.objThreshold)
    
    detect_image = yolonet.detect(color_data)
    elapsed_time = time.time() - starting_time
    cv2.imshow("yolov5 detect img", detect_image)
    print(1/elapsed_time)

    depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_data, alpha=0.03), cv2.COLORMAP_JET)
    cv2.line(color_data, (350, 0), (350, horiBound[0]), (0, 0, 255), 1)
    cv2.line(color_data, (vertiBound[0], horiBound[0]), (vertiBound[0], 480), (0, 0, 255), 1)
    cv2.line(color_data, (vertiBound[1], horiBound[0]), (vertiBound[1], 480), (0, 0, 255), 1)
    cv2.line(color_data, (vertiBound[0], horiBound[0]), (vertiBound[1], horiBound[0]), (0, 0, 255), 1) # 1 m 阈值线

    output = np.hstack([color_data, depth_image])
    
    cv2.imshow("final", output)
    
    cv2.waitKey(1)



class Orb_slam_sub(Node):
    def __init__(self, name):
        super().__init__(name)
        color_frame = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        depth_frame = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        color_depth = message_filters.TimeSynchronizer([color_frame, depth_frame], 1)
        color_depth.registerCallback(self.listener_callback)
        self.cv_bridge = CvBridge()

    def listener_callback(self, color_frame, depth_frame):
        color = self.cv_bridge.imgmsg_to_cv2(color_frame, "bgr8")
        depth = self.cv_bridge.imgmsg_to_cv2(depth_frame, "16UC1")
        processing(color, depth)

def main(args=None):
    rclpy.init(args=args)
    node = Orb_slam_sub("orb_slam_sub")

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown