from matplotlib import patches
import matplotlib.pyplot as plt
import json
import cv2

# 文件路径
image_path = r'C:\\Users\\17802\\AppData\\LocalLow\\DefaultCompany\\TLARC Playground\\solo\\sequence.0\\step0.camera.png'
json_path = r'C:\\Users\\17802\\AppData\\LocalLow\\DefaultCompany\\TLARC Playground\\solo\\sequence.0\\step0.frame_data.json'

# 读取图像
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 读取 JSON 数据
with open(json_path, 'r') as f:
    data_dict = json.load(f)

annotations = data_dict["captures"][0]["annotations"]

# 可视化设置
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image_rgb)
ax.set_xlim(0, image_rgb.shape[1])
ax.set_ylim(image_rgb.shape[0], 0)

# 定义颜色
keypoints_annotations = [ann for ann in annotations if ann["@type"] == "type.unity.com/unity.solo.KeypointAnnotation"]
max_instance_id = max(key["instanceId"] for keypoints in keypoints_annotations for key in keypoints["values"])
cmap = plt.cm.get_cmap('hsv', max_instance_id + 1)

# 绘制 BoundingBox2DAnnotation
bounding_boxes_annotations = [ann for ann in annotations if ann["@type"] == "type.unity.com/unity.solo.BoundingBox2DAnnotation"]

if bounding_boxes_annotations:
    for bbox_annotation in bounding_boxes_annotations:
        for bbox in bbox_annotation.get("values", []):
            if "origin" in bbox and "dimension" in bbox:
                top_left = bbox["origin"]
                width = bbox["dimension"][0]
                height = bbox["dimension"][1]
                rect = patches.Rectangle((top_left[0], top_left[1]), width, height, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

# 绘制关键点和实例 ID
unique_instance_ids = set()
for keypoints in keypoints_annotations:
    keypoints_values = keypoints["values"]
    for keypoints in keypoints_values:
        instance_id = keypoints["instanceId"]
        color = cmap(instance_id)
        if instance_id not in unique_instance_ids:
            label = f'Instance {instance_id}' 
            unique_instance_ids.add(instance_id)
        else:
            label = ""
        for kp in keypoints["keypoints"]:
            x, y = kp["location"]
            ax.scatter(x, y, color=color, label=label)

# 去掉重复的图例标签
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

plt.title("Keypoints and Instance IDs")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
