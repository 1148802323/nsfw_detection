from transformers import pipeline

classifier = pipeline("image-classification", model=r"D:\code\nsfw_detection\train\results\final")

def execute2(image):  # image 为pillow格式
    detection_result = classifier(image)
    print(detection_result)

    # 初始化默认值
    nsfw_score = 0.0
    normal_score = 0.0

    # 从检测结果中提取分数
    for i in detection_result:
        if i.get('label') == 'nsfw':
            nsfw_score = i.get('score')
        elif i.get('label') == 'normal':
            normal_score = i.get('score')

    # 直接返回两个浮点数
    return nsfw_score, normal_score
