import cv2
import os

def Resize_images_main(input_folder, output_folder):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 构建完整的文件路径
        input_path = os.path.join(input_folder, filename)
        # 检查是否为图像文件，这里简单通过扩展名判断
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 读取图像
            image = cv2.imread(input_path)
            if image is not None:
                # 调整图像大小为 256x256
                resized_image = cv2.resize(image, (680, 1560))
                # 构建输出文件的完整路径
                output_path = os.path.join(output_folder, filename.replace('.png', '.jpg'))
                # 保存调整大小后的图像
                cv2.imwrite(output_path, resized_image)
                print(f"已处理并保存: {output_path}")
            else:
                print(f"无法读取图像: {input_path}")
#
# # 示例使用
# input_folder = 'boat'
# output_folder = 'imgs'
# resize_all_images(input_folder, output_folder)