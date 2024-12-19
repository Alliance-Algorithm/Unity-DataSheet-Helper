import os
import shutil
import random
from tqdm import tqdm

# 文件夹路径
folder_path = r'C:\\Users\\17802\\AppData\\LocalLow\\DefaultCompany\\TLARC Playground\\solo_1\\sequence.0'
output_folder_path = os.path.join(folder_path, 'yolo')

# 确定训练和验证集目录
train_img_dir = os.path.join(output_folder_path, 'train', 'images')
train_label_dir = os.path.join(output_folder_path, 'train', 'labels')
val_img_dir = os.path.join(output_folder_path, 'val', 'images')
val_label_dir = os.path.join(output_folder_path, 'val', 'labels')

# 创建输出目录
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 获取所有 step{id}.camera.png 和相应的 step{id}.txt 文件
all_files = [(f, f.replace('.camera.png', '.txt')) for f in os.listdir(folder_path) if f.endswith('.camera.png') and os.path.exists(os.path.join(output_folder_path, f.replace('.camera.png', '.txt')))]

# 随机选取1000个文件对
selected_files = random.sample(all_files, min(65000, len(all_files)))

# 按照7:3的比例分为训练集和验证集
split_idx = int(0.7 * len(selected_files))
train_files = selected_files[:split_idx]
val_files = selected_files[split_idx:]

# 复制文件到训练集和验证集目录 
def copy_files(file_pairs, img_dir, label_dir):
    for img_file, label_file in tqdm(file_pairs, desc=f'Copying files to {img_dir}'):
        shutil.copy(os.path.join(folder_path, img_file), os.path.join(img_dir, img_file.replace('.camera.png', '.png')))
        shutil.copy(os.path.join(output_folder_path, label_file), os.path.join(label_dir, label_file))

copy_files(train_files, train_img_dir, train_label_dir)
copy_files(val_files, val_img_dir, val_label_dir)

print(f"数据集已生成在目录：{output_folder_path}")
