import cv2
import numpy as np
import onnxruntime

# 加载 ONNX 模型
model_path = "export/best.onnx"
session = onnxruntime.InferenceSession(model_path)

# 创建视频捕获对象
video_path = "video/3.mp4"
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 定义类别名称字典
names = {0: 'none', 1: 'class1', 2: 'class2', 3: 'class3', 4: 'class4', 5: 'class5', 6: 'class6'}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        break

    # 预处理图像
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float64) / 255.0).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    # 运行模型推理
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img})

    # 处理输出
    output = outputs[0]
    num_detections = output.shape[2]
    num_keypoints = 6 

    boxes = []
    confidences = []
    class_ids = []
    all_keypoints = []

    for i in range(num_detections):
        confidence = max(output[0, 4:4+7, i])
        if confidence < 0.8:
            continue

        x_center = output[0, 0, i] * frame.shape[1] / 640
        y_center = output[0, 1, i] * frame.shape[0] / 640
        width = output[0, 2, i] * frame.shape[1] / 640
        height = output[0, 3, i] * frame.shape[0] / 640

        x = int(x_center - width / 2)
        y = int(y_center - height / 2)

        boxes.append([x, y, int(width), int(height)])
        confidences.append(float(confidence))
        class_ids.append(np.argmax(output[0, 4:4+7, i]))

        keypoints = []
        for k in range(num_keypoints):
            px = int(output[0, 4 + 7 + 3 * k, i] * frame.shape[1] / 640)
            py = int(output[0, 5 + 7 + 3 * k, i] * frame.shape[0] / 640)
            keypoints.append((px, py))
        all_keypoints.append(keypoints)

    # 执行非极大值抑制
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.8, nms_threshold=0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            label = f"{names[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            keypoints = all_keypoints[i]
            for point in keypoints:
                cv2.circle(frame, point, 5, (255, 0, 0), -1)

    # 显示带有预测结果的视频帧
    cv2.imshow("YOLOPose Video Prediction", frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
