# yolov7-opencv-onnxrun-cpp-py
分别使用OpenCV、ONNXRuntime部署YOLOV7目标检测，一共包含12个onnx模型，依然是包含C++和Python两个版本的程序。

编写这套YOLOV7的程序，跟此前编写的YOLOV6的程序，大部分源码是相同的，区别仅仅在于图片预处理的过程不一样。
YOLOV7的图片预处理是BGR2RGB+不保持高宽比的resize+除以255

由于onnx文件太多，无法直接上传到仓库里，需要从百度云盘下载，
链接: https://pan.baidu.com/s/1FoC0n7qMz4Fz0RtDGpI6xQ  密码: 7mhs
下载完成后把models目录放在主程序文件的目录内，编译运行

YOLOV7的训练源码是 https://github.com/WongKinYiu/yolov7
跟YOLOR是同一个作者的。
