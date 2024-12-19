import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# 加载预训练的 YOLOPose 模型
model = YOLO("./export/best.pt")

# 创建视频捕获对象
video_path = "video/4.mp4"
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    # ret = True
    # frame = cv2.imread(r"solo_1\sequence.0\step18680.camera.png")
    if not ret:
        print("无法读取视频帧")
        break
    # image =  cv2.resize(frame, (640, 640))
    bImg, gImg, rImg = cv2.split(frame) 
    img = cv2.subtract(rImg , bImg)  
    frame = np.zeros_like(frame)  # 创建与 img1 相同形状的黑色图像
    frame[:,:,2] =  img # 在黑色图像模板添加红色分量 rImg
    results = model(frame)

    # 标注预测结果
    annotator = Annotator(frame, line_width=2)
    for result in results:
        boxes = result.boxes
        keypoints = result.keypoints
        for box, kpts in zip(boxes, keypoints):
            # 提取坐标值
            xyxy = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0])
            conf = box.conf[0]  # 获取置信度
            label = f"{cls} {conf:.2f}"  # 创建标签，包含类别和置信度
            annotator.box_label(xyxy, label, color=colors(cls, True))

            # 标注关键点
            for kp in kpts:
               for data in kp.data:
                    #  print(data)
                     for x,y,conf in  data.cpu().numpy(): # 提取前3个值作为x, y和置信度
                        print(x,y,conf)
                        if conf > 0.995:  # 只标注置信度大于0.5的关键点
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # 显示带有预测结果的视频帧
    cv2.imshow("YOLOPose Video Prediction", frame)

   

    # 按 'q' 键退出
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
