import pyrealsense2 as rs
import numpy as np
# 移除 ros 中 Python 路径
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
# 导入包过后重新添加回 ros 中的 Python 路径
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')


if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)
    try:
        while True:
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
            depth_back = np.where(depth_back < 1000, depth_back, depth_back.max())

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
            depth_back_colormap = cv2.applyColorMap(cv2.convertScaleAbs(out_mask, alpha=0.03), cv2.COLORMAP_JET)
            cv2.putText(depth_back_colormap, f"Shape center: ({shape_middle_x},{shape_middle_y})", (depth_back.shape[0]//2, depth_back.shape[1]//2), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(0, 255, 0))

            # Stack both images horizontally
            # images = np.hstack((color_image, depth_colormap, depth_back_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', depth_back_colormap)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()
