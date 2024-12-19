import os
import json
from tqdm import tqdm

# 文件夹路径
folder_path = r'C:\\Users\\17802\\AppData\\LocalLow\\DefaultCompany\\TLARC Playground\\solo_1\\sequence.0'
output_folder_path = os.path.join(folder_path, 'yolo')
image_width = 1280
image_height = 720

# 创建输出文件夹
os.makedirs(output_folder_path, exist_ok=True)

# 遍历文件夹中的所有 JSON 文件
for filename in tqdm(os.listdir(folder_path), desc="Processing JSON files"):
    if filename.endswith('.frame_data.json'):
        json_path = os.path.join(folder_path, filename)
        txt_output_path = os.path.join(output_folder_path, filename.replace('.frame_data.json', '.txt'))

        # 读取 JSON 数据
        with open(json_path, 'r') as f:
            data_dict = json.load(f)

        annotations = data_dict["captures"][0]["annotations"]

        keypoints_annotations = [ann for ann in annotations if ann["@type"] == "type.unity.com/unity.solo.KeypointAnnotation"]
        bounding_boxes_annotations = [ann for ann in annotations if ann["@type"] == "type.unity.com/unity.solo.BoundingBox2DAnnotation"]

        # 创建映射，用于快速查找 BoundingBox2DAnnotation
        bbox_mapping = {}
        for bbox_annotation in bounding_boxes_annotations:
            for bbox in bbox_annotation.get("values", []):
                instance_id = bbox["instanceId"]
                if "labelName" in bbox and bbox["labelName"].startswith("L"):
                    class_index = int(bbox["labelName"][1:])
                    bbox_mapping[instance_id] = (bbox, class_index)

        # 将数据写入 txt 文件
        with open(txt_output_path, 'w') as txt_file:
            for keypoints_annotation in keypoints_annotations:
                keypoints_values = keypoints_annotation["values"]
                for keypoints in keypoints_values:
                    instance_id = keypoints["instanceId"]
                    if instance_id in bbox_mapping:
                        bbox, class_index = bbox_mapping[instance_id]
                        # 检查 "origin" 和 "dimension" 是否存在
                        if "origin" in bbox and "dimension" in bbox:
                            # 标准化 x, y, width, height
                            width = bbox["dimension"][0] / image_width
                            height = bbox["dimension"][1] / image_height
                            x = bbox["origin"][0] / image_width+ width/ 2
                            y = bbox["origin"][1] / image_height + height / 2

                            # 初始化关键点数据
                            keypoint_data = []
                            for kp in keypoints["keypoints"]:
                                px, py = kp["location"]
                                visibility = 1 if kp["state"] != 0 else 0
                                px = px / image_width
                                py = py / image_height
                                keypoint_data.extend([px, py, visibility])

                            # 将所有数据写入文件
                            txt_file.write(f"{class_index} {x} {y} {width} {height} " + " ".join(map(str, keypoint_data)) + "\n")

print(f"YoloPose 数据集已生成在目录：{output_folder_path}")
