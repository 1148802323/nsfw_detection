import os
from obs import ObsClient


def download_images(key_file, output_dir, max_count=2000):
    """
    从华为云的OBS批量下载图片
    (依赖项应该已经在requirements里面了，如果obs未导入，则取官方文档下载即可：https://support.huaweicloud.com/sdk-python-devg-obs/obs_22_0400.html)

    参数:
    key_file: 包含ObjectKey列表的文本文件
    output_dir: 图片保存目录
    max_count: 最大下载数量
    """
    # OBS认证信息，详细见文档：https://support.huaweicloud.com/sdk-python-devg-obs/obs_22_0910.html
    ak = ''
    sk = ''
    server = ""
    bucketName = ''

    # 创建OBS客户端实例
    obs_client = ObsClient(access_key_id=ak, secret_access_key=sk, server=server)

    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")

    # 读取ObjectKey列表
    try:
        with open(key_file, 'r', encoding='utf-8') as f:
            object_keys = [line.strip() for line in f.readlines() if line.strip()] # 读的是换行符
    except Exception as e:
        print(f"读取文件 {key_file} 时出错: {e}")
        obs_client.close()
        return

    # 限制下载数量
    if len(object_keys) > max_count:
        object_keys = object_keys[:max_count]

    print(f"开始下载 {len(object_keys)} 个图片...")

    # 下载计数器
    success_count = 0
    fail_count = 0

    # 逐个下载图片
    for i, object_key in enumerate(object_keys, 1):
        # 生成输出文件名（使用ObjectKey作为文件名）
        # 注：ObjectKey可能包含路径分隔符，需要处理
        safe_filename = object_key.replace('/', '_').replace('\\', '_')
        download_path = os.path.join(output_dir, safe_filename)

        print(f"下载进度: {i}/{len(object_keys)} - {object_key}")

        try:
            # 下载图片
            resp = obs_client.getObject(
                bucketName=bucketName,
                objectKey=object_key,
                downloadPath=download_path
            )

            if resp.status < 300:
                success_count += 1
                print(f"成功下载: {object_key}")
            else:
                fail_count += 1
                print(f"下载失败: {object_key} - 错误代码: {resp.errorCode}")

        except Exception as e:
            fail_count += 1
            print(f"下载异常: {object_key} - 错误信息: {str(e)}")

    # 关闭OBS客户端
    obs_client.close()

    # 输出下载结果
    print(f"下载完成! 成功: {success_count}, 失败: {fail_count}")


if __name__ == "__main__":
    # 正常图片
    key_file = "porn_object_keys_2000.txt"  # 包含ObjectKey列表的文件
    output_dir = "./porn_images_2000"  # 图片保存目录
    # # porn图片
    # key_file = "porn_object_keys.txt"
    # output_dir = "./downloaded_images"

    max_count = 2000  # 最大下载数量

    # 执行下载
    download_images(key_file, output_dir, max_count)