import os
from PIL import Image

def crop_image(input_path, output_path, left=100, top=100, right=0, bottom=0):
    """
    裁剪图像以去除左上角文字信息，并将图片向右旋转90度

    参数:
    input_path (str): 输入图像路径
    output_path (str): 输出图像路径
    left (int): 左边界裁剪像素数
    top (int): 上边界裁剪像素数
    right (int): 右边界保留像素数(0表示到边缘)
    bottom (int): 下边界保留像素数(0表示到边缘)
    """
    try:
        # 打开图像
        img = Image.open(input_path)
        width, height = img.size

        # 计算裁剪区域
        cropped_left = left
        cropped_top = top
        cropped_right = width - right if right > 0 else width
        cropped_bottom = height - bottom if bottom > 0 else height

        # 执行裁剪
        cropped_img = img.crop((cropped_left, cropped_top, cropped_right, cropped_bottom))

        # 向右旋转90度
        rotated_img = cropped_img.rotate(-90, expand=True)

        # 保存结果
        rotated_img.save(output_path)
        print(f"裁剪并旋转后的图像已保存至: {output_path}")

    except Exception as e:
        print(f"处理图像时出错: {e}")


def crop_images_in_folder(input_folder, output_folder, left=100, top=100, right=0, bottom=0):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            crop_image(input_path, output_path, left, top, right, bottom)


def Crop_images_main(input_folder, output_folder, left=150, top=80):
    crop_images_in_folder(input_folder, output_folder, left=left, top=top)


if __name__ == "__main__":
    # 使用示例 - 需要替换为实际文件夹路径
    input_folder = "data2"  # 输入图片文件夹路径
    output_folder = "output_images"  # 输出图片文件夹路径

    # 调整裁剪参数以获得最佳效果
    crop_images_in_folder(input_folder, output_folder, left=150, top=80)