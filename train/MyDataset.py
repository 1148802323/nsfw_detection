import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from transformers import ViTImageProcessor
from PIL import Image

"""
此文件仅为练手版本，已废弃，仅供参考

"""

class NSFWDataset(Dataset):
    def __init__(self, dataset_path, split, model_name="nsfw_image_detection"):
        """
        废弃版NSFW数据集

        Args:
            dataset_path (str): 数据集路径
            split (str): 数据集拆分，可选 'train', 'validation', 'test'
            model_name (str): 模型名称，用于获取正确的图像处理器
        """
        # 加载数据集
        self.dataset = load_from_disk(dataset_path)
        print(self.dataset)

        # 选择正确的拆分
        if split in self.dataset:
            self.dataset = self.dataset[split]
        else:
            raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(self.dataset.keys())}")

        # 加载图像处理器
        self.processor = ViTImageProcessor.from_pretrained(model_name)

        # 检查数据集是否包含必要的字段
        if 'image' not in self.dataset.column_names:
            raise ValueError("数据集要有 'image' 列")
        if 'label' not in self.dataset.column_names:
            raise ValueError("数据集要有 'label' 列")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # 获取图像
        image = item['image']

        # 如果图像是文件路径，则加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        # 使用处理器处理图像
        processed_image = self.processor(image, return_tensors="pt")["pixel_values"].squeeze()

        # 获取标签
        label = item['label']

        return {
            'pixel_values': processed_image,
            'labels': torch.tensor(label, dtype=torch.long)
        }

if __name__ == '__main__':
    NsfwDataset = NSFWDataset(dataset_path=r"D:\code\nsfw_detection\train\set_data\dataset")