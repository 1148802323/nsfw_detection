import os
import re
import glob
from datetime import datetime, timedelta


def extract_normal_object_keys(log_directory, output_file, start_index=None, end_index=None, max_count=None):
    """
    从日志文件中提取Suggestion为pass的ObjectKey

    参数:
    log_directory: 日志文件所在的目录
    output_file: 输出文件路径
    start_index: 起始索引（从0开始），如果为None则从第一个开始
    end_index: 结束索引（包含），如果为None则提取到max_count或文件末尾
    max_count: 最大提取数量，如果为None则使用范围模式
    """
    # 存储提取的ObjectKey
    object_keys = []
    count = 0
    total_detected = 0  # 总共检测到的正常图片数量

    # 参数验证
    if max_count is not None and (start_index is not None or end_index is not None):
        print("警告: max_count模式与范围模式同时存在，优先使用max_count模式")
        start_index = None
        end_index = None

    # 获取所有日志文件
    log_files = glob.glob(os.path.join(log_directory, "*.log"))
    # log_files.sort()  # 按文件名排序，即按时间顺序

    # 正则表达式模式 - 匹配包含ObjectKey和Suggestion的完整行
    pattern = re.compile(r'recive http post.*ObjectKey: (\S+).*Suggestion: (\S+)')

    print(f"找到 {len(log_files)} 个日志文件")

    # 根据模式显示不同的提示信息
    if max_count is not None:
        print(f"开始提取最多 {max_count} 个正常图片ObjectKey...")
    elif start_index is not None or end_index is not None:
        start = start_index if start_index is not None else 0
        end = end_index if end_index is not None else "末尾"
        print(f"开始提取第 {start} 到第 {end} 个检测结果...")
    else:
        print("开始提取所有正常图片ObjectKey...")

    # 遍历所有日志文件
    for log_file in log_files:
        if (max_count is not None and len(object_keys) >= max_count) or \
                (end_index is not None and count > end_index):
            break

        print(f"处理文件: {os.path.basename(log_file)}")

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    # 检查停止条件
                    if (max_count is not None and len(object_keys) >= max_count) or \
                            (end_index is not None and count > end_index):
                        break

                    # 查找包含ObjectKey和Suggestion的行
                    if 'recive http post' in line and 'ObjectKey:' in line and 'Suggestion:' in line:
                        match = pattern.search(line)
                        if match:
                            object_key = match.group(1)
                            # 移除可能存在的逗号（由于下载代码读取的是换行符）
                            if object_key.endswith(','):
                                object_key = object_key[:-1]
                            suggestion = match.group(2)

                            if suggestion == 'pass':
                                total_detected += 1

                                # 根据模式决定是否添加到结果
                                should_add = False

                                if max_count is not None:
                                    # max_count模式：只要还没达到最大数量就添加
                                    should_add = len(object_keys) < max_count
                                elif start_index is not None and end_index is not None:
                                    # 范围模式：在指定范围内才添加
                                    should_add = start_index <= count <= end_index
                                elif start_index is not None:
                                    # 只有起始索引：从start_index开始到末尾
                                    should_add = count >= start_index
                                elif end_index is not None:
                                    # 只有结束索引：从开始到end_index
                                    should_add = count <= end_index
                                else:
                                    # 无限制模式：全部添加
                                    should_add = True

                                if should_add:
                                    object_keys.append(object_key)
                                    print(f"找到第 {count} 个正常图片: {object_key}")

                                count += 1

        except Exception as e:
            print(f"处理文件 {log_file} 时出错: {e}")

    # 将结果写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for key in object_keys:
            f.write(f"{key}\n")

    print(f"完成! 总共检测到 {total_detected} 个正常图片")
    print(f"提取了 {len(object_keys)} 个ObjectKey，已保存到 {output_file}")

    # 显示提取范围信息
    if max_count is not None:
        print(f"提取模式: 最多 {max_count} 个")
    elif start_index is not None or end_index is not None:
        start = start_index if start_index is not None else 0
        end = end_index if end_index is not None else count - 1
        print(f"提取范围: 第 {start} 到第 {min(end, count - 1)} 个检测结果")
    else:
        print("提取模式: 所有检测结果")


if __name__ == "__main__":
    # 配置参数
    log_directory = r"D:\code\plugs\nsfw\showimageserver_logs_20250822\logs\SO3ShowImageServer"
    output_file = "normal_object_keys_2000.txt"

    # 使用示例1: max_count模式（提取最多2000个）
    # extract_normal_object_keys(log_directory, output_file, max_count=2000)

    # 使用示例2: 范围模式（提取第100到第300个）
    # extract_normal_object_keys(log_directory, output_file, start_index=100, end_index=300)

    # 使用示例3: 从某个索引开始到末尾
    # extract_normal_object_keys(log_directory, output_file, start_index=500)

    # 使用示例4: 从开始到某个索引结束
    # extract_normal_object_keys(log_directory, output_file, end_index=1000)

    # 使用示例5: 提取所有
    # extract_normal_object_keys(log_directory, output_file)

    extract_normal_object_keys(log_directory, output_file, start_index=2000, end_index=2200)