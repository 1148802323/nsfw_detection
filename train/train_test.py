import os
import logging
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from datasets import load_from_disk
import numpy as np
import torch
import evaluate
from PIL import Image
from torch.utils.data import Dataset

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_model_and_processor(model_path):
    """设置模型和处理器"""
    try:
        # 加载模型
        model = ViTForImageClassification.from_pretrained(
            model_path,
            num_labels=2,
            id2label={0: "normal", 1: "nsfw"},
            label2id={"normal": 0, "nsfw": 1},
            ignore_mismatched_sizes=True
        )

        # 加载处理器
        processor = ViTImageProcessor.from_pretrained(model_path)

        return model, processor
    except Exception as e:
        logger.error(f"加载模型和处理器时出错: {e}")
        raise


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 路径设置
    model_path = r"D:\code\nsfw_detection\nsfw_image_detection"
    dataset_path = r"D:\code\nsfw_detection\train\set_data\dataset"
    output_dir = r"D:\code\nsfw_detection\train\results"

    # 检查路径是否存在
    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        return

    if not os.path.exists(dataset_path):
        logger.error(f"数据集路径不存在: {dataset_path}")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 设置模型和处理器
        model, processor = setup_model_and_processor(model_path)
        model.to(device)

        # 加载数据集
        dataset = load_from_disk(dataset_path)
        logger.info(f"数据集加载成功: {dataset}")

        # 定义数据集类
        # 定义数据集类
        class NSFWDataset(Dataset):
            def __init__(self, dataset, split="train"):
                """
                初始化NSFW数据集
                这一部分可以封装到其他py文件中，由于是测试train，所以就没分开

                Args:
                    dataset: 可以是 DatasetDict 或数据集路径字符串
                    split (str): 数据集拆分，可选 'train', 'validation', 'test'
                    processor: 图像处理器 (这个是nsfw_image_detection模型包里面写的一个preprocessor_config.json文件，
                    里面是一些图像大小和格式的处理，可以根据需求修改)
                    ！！！！
                    注，这个测试训练的输出中不包含这个json文件，如果需要用训练后的模型做测试在修改完模型路径后记得复制一份到模型文件夹里！要不会报错！
                """
                # 如果传入的是字符串路径，则加载数据集
                if isinstance(dataset, str):
                    self.dataset = load_from_disk(dataset)
                else:
                    self.dataset = dataset  # 假设已经是 DatasetDict 对象

                # 选择正确的拆分
                if split in self.dataset:
                    self.dataset = self.dataset[split]
                else:
                    raise ValueError(
                        f"Split '{split}' not found in dataset. Available splits: {list(self.dataset.keys())}")

                self.processor = processor

                # 移除不需要的列
                columns_to_remove = [col for col in self.dataset.column_names if col not in ['image', 'label']]
                if columns_to_remove:
                    self.dataset = self.dataset.remove_columns(columns_to_remove)

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                item = self.dataset[idx]

                # 获取图像
                image_data = item['image']

                # 检查 image_data 的类型
                if isinstance(image_data, str):
                    # 如果是字符串，则假设是图像路径并加载图像
                    image = Image.open(image_data).convert('RGB')
                elif hasattr(image_data, 'mode'):
                    # 如果已经是 PIL 图像对象，直接使用
                    image = image_data
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                else:
                    # 其他情况，尝试转换为 PIL 图像
                    try:
                        image = Image.fromarray(image_data).convert('RGB')
                    except:
                        raise ValueError(f"无法处理图像数据: {type(image_data)}")

                # 使用处理器处理图像
                inputs = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs['pixel_values'].squeeze(0)  # 移除批次维度

                # 获取标签
                label = item['label']

                return {
                    'pixel_values': pixel_values,
                    'labels': torch.tensor(label, dtype=torch.long)
                }

        # 创建训练和验证数据集
        train_dataset = NSFWDataset(dataset, split="train")
        val_dataset = NSFWDataset(dataset, split="validation")

        logger.info(f"训练集大小: {len(train_dataset)}")
        logger.info(f"验证集大小: {len(val_dataset)}")

        # 定义训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            eval_strategy="epoch",  # 每个 epoch 结束后进行评估
            save_strategy="epoch",  # 每个 epoch 结束后保存模型
            num_train_epochs=10,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True, # 这个是保存最终训练效果为最佳的的模型
            metric_for_best_model="accuracy",
            greater_is_better=True,
            push_to_hub=False,
        )

        # 定义评估指标
        metric = evaluate.load("accuracy") #这个是hugging face支持的评估函数

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        # 创建Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        # 训练
        logger.info("开始训练...")
        trainer.train()

        # 保存最终模型
        final_model_path = os.path.join(output_dir, "final")
        trainer.save_model(final_model_path)
        logger.info(f"训练完成，模型已保存到: {final_model_path}")

    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        raise


if __name__ == "__main__":
    main()

