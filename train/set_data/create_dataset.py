import os
from PIL import Image
from datasets import Dataset, DatasetDict, Image
import pandas as pd
from sklearn.model_selection import train_test_split


def create_nsfw_dataset(nsfw_image_folder, normal_image_folder, output_path, test_size=0.2, val_size=0.1):
    """
    从NSFW图片文件夹和正常图片文件夹创建带标签的数据集

    Args:
        nsfw_image_folder (str): 包含NSFW图片的文件夹路径
        normal_image_folder (str): 包含正常图片的文件夹路径
        output_path (str): 保存数据集的路径
        test_size (float): 测试集比例
        val_size (float): 验证集比例
    """
    # 收集NSFW图片路径
    nsfw_image_paths = []
    for img_name in os.listdir(nsfw_image_folder):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            nsfw_image_paths.append(os.path.join(nsfw_image_folder, img_name))

    # 收集正常图片路径
    normal_image_paths = []
    for img_name in os.listdir(normal_image_folder):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            normal_image_paths.append(os.path.join(normal_image_folder, img_name))

    # 创建标签列表 (NSFW=1, 正常=0)
    nsfw_labels = [1] * len(nsfw_image_paths)
    normal_labels = [0] * len(normal_image_paths)

    # 合并路径和标签
    all_image_paths = nsfw_image_paths + normal_image_paths
    all_labels = nsfw_labels + normal_labels

    # 创建数据框
    df = pd.DataFrame({
        'image': all_image_paths,  # 存储路径而不是图像对象
        'label': all_labels
    })

    # 划分训练集、验证集和测试集
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(train_df, test_size=val_size/(1-test_size), random_state=42, stratify=train_df['label'])

    # 创建数据集
    train_dataset = Dataset.from_pandas(train_df).cast_column("image", Image())
    val_dataset = Dataset.from_pandas(val_df).cast_column("image", Image())
    test_dataset = Dataset.from_pandas(test_df).cast_column("image", Image())

    # 创建DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

    # 保存数据集
    dataset_dict.save_to_disk(output_path)
    print(f"数据集已保存到 {output_path}")
    print(f"训练集: {len(train_dataset)} 张图片")
    print(f"验证集: {len(val_dataset)} 张图片")
    print(f"测试集: {len(test_dataset)} 张图片")
    print(f"NSFW图片数量: {len(nsfw_image_paths)}")
    print(f"正常图片数量: {len(normal_image_paths)}")
    print(f"正常图片比例: {len(normal_image_paths)/(len(all_image_paths)):.2%}")

    return dataset_dict


# 使用示例
if __name__ == '__main__':
    nsfw_image_folder = r"D:\code\nsfw_detection\downloaded_images"  # NSFW图片文件夹路径
    normal_image_folder = r"D:\code\nsfw_detection\pass_images"  # 正常图片文件夹路径
    output_path = r"D:\code\nsfw_detection\train\set_data\dataset"  # 数据集保存路径
    dataset = create_nsfw_dataset(nsfw_image_folder, normal_image_folder, output_path)