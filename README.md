# ZT
ZT on kuavo
# checkpoints
The models should be placed in the `checkpoints` folder.
- "BAAI/bge-small-zh-v1.5".
- rcnndnn. 
- yolov5weight.
- vgg_face. The model can be got reference to this [repository](https://github.com/serengil/deepface) on github. This model will be downloaded defaultly in '~/.deepface/weight/'
- bert. The base model is finetuned to caculate ppl.
# Acknowledgements:
We would like to extend our heartfelt gratitude to Leju company, a distinguished entity based in Shenzhen, China. Their unwavering support has been instrumental in the successful completion of this project.


# how to launch realsense for noetic
we need to cmake it. However leju prepared it before( need to pull it from lejuhub.com )

cd ~/kuavo_ros_application
. ./devel_isolated/setup.sh
roslaunch kuavo_robot_ros sensor_robot_enable.launch 