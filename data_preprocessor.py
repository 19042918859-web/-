import os
import cv2
import numpy as np
import json
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE
import albumentations as A
from albumentations.pytorch import ToTensorV2

print("=== 步骤1: 数据预处理 ===")

# 配置参数
DATA_PATH = 'Garbage classification'
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic']  # 5个类别
IMG_SIZE = (224, 224)

# 定义数据增强
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),  # 替换 ShiftScaleRotate
    A.RandomBrightnessContrast(p=0.3),
    A.HueSaturationValue(p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.3),  # 添加 fill_value
])


# 转换numpy类型为Python原生类型的函数
def convert_to_python_types(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    else:
        return obj


def augment_image(image, num_augmentations=3):
    """对图像进行数据增强"""
    augmented_images = []
    for _ in range(num_augmentations):
        augmented = augmentation(image=image)
        augmented_images.append(augmented['image'])
    return augmented_images


# 1. 验证数据
print("1. 验证数据集...")
class_stats = {}
for class_name in CLASSES:
    class_path = os.path.join(DATA_PATH, class_name)
    if os.path.exists(class_path):
        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        class_stats[class_name] = len(images)
        print(f"   {class_name}: {len(images)} 张图像")
    else:
        print(f"   ⚠️ 缺失目录: {class_name}")

# 2. 加载和预处理图像
print("2. 加载和预处理图像...")
all_images = []
all_labels = []

for class_idx, class_name in enumerate(CLASSES):
    class_path = os.path.join(DATA_PATH, class_name)
    if not os.path.exists(class_path):
        continue

    images = [f for f in os.listdir(class_path)
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"  处理 {class_name} 类图像...")
    
    for img_name in images:
        img_path = os.path.join(class_path, img_name)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)
                
                # 基本预处理
                img_normalized = img.astype(np.float32) / 255.0
                all_images.append(img_normalized)
                all_labels.append(class_idx)
                
                # 对小样本类别进行数据增强
                if class_stats[class_name] < 450:  # 如果样本少于450张
                    augmented_imgs = augment_image(img)
                    for aug_img in augmented_imgs:
                        aug_img_resized = cv2.resize(aug_img, IMG_SIZE)
                        aug_img_normalized = aug_img_resized.astype(np.float32) / 255.0
                        all_images.append(aug_img_normalized)
                        all_labels.append(class_idx)
                        
        except Exception as e:
            print(f"    处理图像 {img_name} 时出错: {str(e)}")
            continue

print(f"成功加载 {len(all_images)} 张有效图像（包含增强图像）")

if len(all_images) == 0:
    print("错误: 没有找到任何有效图像!")
    exit(1)

# 3. 检查类别平衡
print("3. 检查类别平衡...")
label_counter = Counter(all_labels)
print("增强后的类别分布:")
for class_idx, count in label_counter.items():
    print(f"  {CLASSES[class_idx]}: {count} 张图像")

# 4. 划分数据集
print("4. 划分训练/验证/测试集...")
X = np.array(all_images)
y = np.array(all_labels)

# 使用分层抽样确保各类别比例一致
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y  # 减小测试集比例
)

# 再从剩余数据中分验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
)

print(f"训练集: {len(X_train)} 样本")
print(f"验证集: {len(X_val)} 样本") 
print(f"测试集: {len(X_test)} 样本")

# 5. 保存数据
print("5. 保存处理后的数据...")
os.makedirs('data/processed/splits', exist_ok=True)

# 保存为numpy文件
np.save('data/processed/splits/X_train.npy', X_train)
np.save('data/processed/splits/y_train.npy', y_train)
np.save('data/processed/splits/X_val.npy', X_val)
np.save('data/processed/splits/y_val.npy', y_val)
np.save('data/processed/splits/X_test.npy', X_test)
np.save('data/processed/splits/y_test.npy', y_test)

# 6. 保存数据集信息
print("6. 保存数据集信息...")
dataset_info = {
    'classes': CLASSES,
    'sizes': {
        'train': int(len(X_train)),
        'val': int(len(X_val)),
        'test': int(len(X_test)),
        'total': int(len(X))
    },
    'class_distribution': {
        'original': {class_name: int(count) for class_name, count in class_stats.items()},  # 修复这里
        'augmented': {
            'train': {CLASSES[k]: int(v) for k, v in Counter(y_train).items()},  # 这里也建议修改
            'val': {CLASSES[k]: int(v) for k, v in Counter(y_val).items()},      # 这里也建议修改
            'test': {CLASSES[k]: int(v) for k, v in Counter(y_test).items()}     # 这里也建议修改
        }
    },
    'image_size': list(IMG_SIZE),
    'random_state': 42,
    'preprocessing_info': {
        'normalization': '0-1',
        'color_space': 'RGB',
        'data_augmentation': True,
        'augmentation_techniques': [
            'HorizontalFlip', 'RandomRotate90', 'ShiftScaleRotate',
            'RandomBrightnessContrast', 'HueSaturationValue',
            'GaussianBlur', 'CoarseDropout'
        ]
    },
    'split_ratios': {
        'train': 0.70,
        'val': 0.15,
        'test': 0.15
    }
}

# 使用转换函数确保所有类型正确
dataset_info = convert_to_python_types(dataset_info)

with open('data/processed/dataset_info.json', 'w', encoding='utf-8') as f:
    json.dump(dataset_info, f, indent=2, ensure_ascii=False)

print("数据预处理完成!")
print(f"\n最终数据集统计:")
print(f"训练集: {len(X_train)} 样本")
print(f"验证集: {len(X_val)} 样本")
print(f"测试集: {len(X_test)} 样本")

# 打印类别分布
print("\n增强后类别分布:")
for split_name, distribution in dataset_info['class_distribution']['augmented'].items():
    print(f"  {split_name}: {distribution}")

# 计算类别平衡性
train_dist = dataset_info['class_distribution']['augmented']['train']
max_count = max(train_dist.values())
min_count = min(train_dist.values())
balance_ratio = min_count / max_count

print(f"\n类别平衡性: {balance_ratio:.2f} ({'良好' if balance_ratio > 0.7 else '一般' if balance_ratio > 0.5 else '需要改进'})")

print("\n数据预处理完成! ✅")