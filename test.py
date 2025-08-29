import shutil

from detection import execute2
import base64
import os
from io import BytesIO
from PIL import Image
import glob


def test_single_image(image_path):
    """测试单张图片"""
    try:
        # 读取图片文件
        with open(image_path, "rb") as image_file:
            # 将图片转换为base64编码
            image_b64 = base64.b64encode(image_file.read()).decode('utf-8')

        # base64转bytes
        byte_data = base64.b64decode(image_b64)
        # 将二进制转为PIL格式图片
        image = Image.open(BytesIO(byte_data))

        # 执行检测
        result_dic = execute2(image)

        return result_dic

    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {str(e)}")
        return None


def test_images_in_folder(folder_path):
    """测试文件夹中的所有JPG图片"""
    # 获取所有图片文件
    image_paths = glob.glob(os.path.join(folder_path, "*.png"))
    image_paths.extend(glob.glob(os.path.join(folder_path, "*.jpg")))
    image_paths.extend(glob.glob(os.path.join(folder_path, "*.jpeg")))
    image_paths.extend(glob.glob(os.path.join(folder_path, "*.bmp")))

    if not image_paths:
        print(f"在文件夹 {folder_path} 中未找到图片")
        return None

    print(f"找到 {len(image_paths)} 张图片，开始测试...")
    print("-" * 50)

    # 创建保存图片的文件夹
    nsfw_folder = os.path.join(folder_path, "nsfw_images")
    normal_folder = os.path.join(folder_path, "normal_images")

    os.makedirs(nsfw_folder, exist_ok=True)
    os.makedirs(normal_folder, exist_ok=True)

    results = {}

    # 统计结果
    total_count = 0
    nsfw_count = 0
    normal_count = 0

    for i, image_path in enumerate(image_paths, 1):
        print(f"正在处理第 {i}/{len(image_paths)} 张图片: {os.path.basename(image_path)}")

        result_nsfw, result_normal = test_single_image(image_path)

        if result_nsfw is not None:
            results[image_path] = result_nsfw
            print(f"NSFW分数: {result_nsfw:.6f}, Normal分数: {result_normal:.6f}")

            # 根据NSFW分数分类保存图片
            if result_nsfw > 0.5:
                nsfw_count += 1
                # 复制图片到nsfw文件夹
                dest_path = os.path.join(nsfw_folder, os.path.basename(image_path))
                shutil.copy2(image_path, dest_path)
                print(f"图片已保存到NSFW文件夹: {dest_path}")
            else:
                normal_count += 1
                # 复制图片到normal文件夹
                dest_path = os.path.join(normal_folder, os.path.basename(image_path))
                shutil.copy2(image_path, dest_path)
                print(f"图片已保存到Normal文件夹: {dest_path}")

            total_count += 1
        else:
            results[image_path] = "处理失败"
            print("处理失败")

        print("-" * 30)

    # 打印统计信息
    print(f"\n处理完成！")
    print(f"总图片数: {total_count}")
    print(f"NSFW图片数: {nsfw_count} ({(nsfw_count / total_count * 100):.2f}%)")
    print(f"Normal图片数: {normal_count} ({(normal_count / total_count * 100):.2f}%)")

    return results


def save_results(results, output_file="test_results.txt"):
    """保存测试结果到文件"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("图片检测结果汇总\n")
        f.write("=" * 50 + "\n\n")

        for image_path, result in results.items():
            f.write(f"图片: {os.path.basename(image_path)}\n")
            f.write(f"完整路径: {image_path}\n")
            if result != "处理失败":
                f.write(f"NSFW分数: {result:.6f}\n")
            else:
                f.write(f"检测结果: {result}\n")
            f.write("-" * 40 + "\n")

    print(f"结果已保存到 {output_file}")




if __name__ == "__main__":
    # 使用方法1：测试单张图片
    # single_image_path = r"D:\code\nsfw_detection\downloaded_images\1735661205_306244774673770146_0.png"
    # result_nsfw, result_normal = test_single_image(single_image_path)
    # print(f"单张图片检测为nsfw结果: {result_nsfw}")

    """
    注:
        处理完这些图片后，会在folder_path创建两个文件夹来存放检测出来正常和nsfw的图片
    """
    # 使用方法2：测试整个文件夹中的图片
    folder_path = r"D:\code\nsfw_detection\test_images"
    results = test_images_in_folder(folder_path)

    # 保存方法2结果
    if results:
        save_results(results)

        # 打印汇总信息
        success_count = sum(1 for result in results.values() if result != "处理失败")
        print(f"\n测试完成！成功处理: {success_count}/{len(results)} 张图片")
