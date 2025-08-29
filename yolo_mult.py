import os
import shutil
import numpy as np
from PIL import Image
import onnxruntime as ort
import json


def predict_with_yolov9(image_path, model_path, labels_path, input_size):
    """
    基于nsfw_image_detection模型包的readme，修改了下需求
    使用YOLOv9模型（Falconsai作者提供的）对单张图片进行预测

    Args:
        image_path (str): 输入图片路径
        model_path (str): ONNX模型文件路径
        labels_path (str): 标签JSON文件路径
        input_size (tuple): 模型期望的输入尺寸(高度, 宽度)

    Returns:
        str: 预测的类别标签
        PIL.Image.Image: 原始图片对象
    """

    def load_json(file_path):
        with open(file_path, "r", encoding='utf-8') as f:
            return json.load(f)

    # 加载标签
    labels = load_json(labels_path)

    # 预处理图片
    original_image = Image.open(image_path).convert("RGB")
    image_resized = original_image.resize(input_size, Image.Resampling.BILINEAR)
    image_np = np.array(image_resized, dtype=np.float32) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))  # [C, H, W]
    input_tensor = np.expand_dims(image_np, axis=0).astype(np.float32)

    # 加载YOLOv9模型
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 运行推理
    outputs = session.run([output_name], {input_name: input_tensor})
    predictions = outputs[0]

    # 后处理预测结果
    predicted_index = np.argmax(predictions)
    predicted_label = labels[str(predicted_index)]

    return predicted_label, original_image


def process_folder(input_folder, output_base, model_path, labels_path, input_size):
    """
    处理整个文件夹中的图片，并根据预测结果分类

    Args:
        input_folder (str): 包含待处理图片的文件夹路径
        output_base (str): 输出文件夹的基础路径
        model_path (str): ONNX模型文件路径
        labels_path (str): 标签JSON文件路径
        input_size (tuple): 模型期望的输入尺寸
    """
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    # 创建输出文件夹
    normal_folder = os.path.join(output_base, "normal")
    nsfw_folder = os.path.join(output_base, "nsfw")
    os.makedirs(normal_folder, exist_ok=True)
    os.makedirs(nsfw_folder, exist_ok=True)

    # 统计信息
    stats = {"normal": 0, "nsfw": 0, "errors": 0}

    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # 检查是否为图片文件
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(filename)[1].lower()
        if ext not in image_extensions:
            continue

        try:
            # 进行预测
            predicted_label, image = predict_with_yolov9(file_path, model_path, labels_path, input_size)

            # 确定目标文件夹
            if predicted_label.lower() == "normal":
                target_folder = normal_folder
                stats["normal"] += 1
            elif predicted_label.lower() == "nsfw":
                target_folder = nsfw_folder
                stats["nsfw"] += 1
            else:
                print(f"未知标签: {predicted_label} for {filename}")
                stats["errors"] += 1
                continue

            # 复制文件到目标文件夹
            target_path = os.path.join(target_folder, filename)
            shutil.copy2(file_path, target_path)

            print(f"已处理: {filename} -> {predicted_label}")

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
            stats["errors"] += 1

    # 打印统计信息
    print("\n处理完成!")
    print(f"正常图片: {stats['normal']}")
    print(f"NSFW图片: {stats['nsfw']}")
    print(f"错误数量: {stats['errors']}")


def main():
    # 路径和参数设置
    input_folder = r"D:\code\nsfw_detection\downloaded_images"  # 输入文件夹路径
    output_base = r"D:\code\nsfw_detection\classified_images"  # 输出基础文件夹路径
    model_path = r"D:\code\nsfw_detection\nsfw_image_detection\falconsai_yolov9_nsfw_model_quantized.pt"  # 模型路径
    labels_path = r"D:\code\nsfw_detection\nsfw_image_detection\labels.json"  # 标签文件路径
    input_size = (224, 224)  # 输入尺寸

    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 - {model_path}")
        return

    if not os.path.exists(labels_path):
        print(f"错误: 标签文件不存在 - {labels_path}")
        return

    if not os.path.exists(input_folder):
        print(f"错误: 输入文件夹不存在 - {input_folder}")
        return

    # 处理文件夹
    process_folder(input_folder, output_base, model_path, labels_path, input_size)


if __name__ == "__main__":
    main()