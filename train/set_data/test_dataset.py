import os
import random
from datasets import load_from_disk
from PIL import Image
import matplotlib.pyplot as plt

"""
注，最后可视化部分在linux下直接注释掉即可（或者修改），其他功能正常
"""
def validate_dataset(dataset_path):
    """
    验证数据集的结构和内容
    """
    # 加载数据集
    dataset = load_from_disk(dataset_path)

    print("数据集结构:")
    print(dataset)

    # 检查每个拆分
    for split in ['train', 'validation', 'test']:
        print(f"\n=== {split.upper()} 拆分 ===")

        # 获取拆分
        split_data = dataset[split]

        # 检查特征
        print(f"特征: {split_data.column_names}")

        # 检查样本数量
        print(f"样本数量: {len(split_data)}")

        # 检查标签分布
        labels = split_data['label']
        nsfw_count = sum(1 for label in labels if label == 1)
        normal_count = sum(1 for label in labels if label == 0)

        print(f"NSFW 样本: {nsfw_count} ({nsfw_count / len(labels) * 100:.2f}%)")
        print(f"正常样本: {normal_count} ({normal_count / len(labels) * 100:.2f}%)")

        # 检查前几个样本
        print("\n前5个样本:")
        for i in range(min(5, len(split_data))):
            sample = split_data[i]
            print(f"  样本 {i}: 标签={sample['label']}, 图像类型={type(sample['image'])}")

            # 如果是文件路径，检查文件是否存在
            if isinstance(sample['image'], str):
                if os.path.exists(sample['image']):
                    print(f"      图像路径有效: {sample['image']}")
                else:
                    print(f"      警告: 图像路径无效: {sample['image']}")

    # 可视化一些样本
    print("\n=== 可视化样本 ===")
    visualize_dataset_samples(dataset['train'])


def visualize_dataset_samples(dataset, num_samples=5):
    """
    可视化数据集中的样本（随机五个）
    """

    random_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

    for i, idx in enumerate(random_indices):
        sample = dataset[idx]
        image = sample['image']

        # 如果图像是文件路径，则加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        # 显示图像
        axes[i].imshow(image)
        axes[i].set_title(f"Label: {sample['label']} (Index: {idx})")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == '__main__':
    dataset_path = r"D:\code\nsfw_detection\train\set_data\dataset" # 数据集路径
    validate_dataset(dataset_path)