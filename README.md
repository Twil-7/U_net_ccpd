# U_net_ccpd

# 环境配置：

python == 3.6

keras == 2.1.5

tensorflow == 1.6.0

opencv == 3.4.2

# 运行：

直接运行main.py，其调用了数据集划分、模型创建、训练、检测等py文件。

crop_ccpd数据集下载：https://blog.csdn.net/Twilight737?spm=1018.2226.3001.5343&type=download

car.mp4：自己随手拍摄的车牌视频流。

detect_video.mp4：语义分割检测出的视频效果。

test_demo：ccpd数据集中测试集的检测效果。

video_demo：视频流的图片集检测效果。

# 实验效果：

经过测试，可在室外实际场景下使用，CPU上单帧耗时150ms，检测精度几近100%。

对场景的要求：车牌在整张图片的占比不能过小，面积占比1/8 - 1/2均可。
