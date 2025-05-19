import os
import threading

# 定义两个文件夹路径
folder1 = 'folder1'
folder2 = 'folder2'

# 检查文件夹是否存在
if not os.path.exists(folder1) or not os.path.exists(folder2):
    print("指定的文件夹不存在，请检查路径。")
else:
    # 获取 folder1 中的所有图片文件（假设为 .jpg 格式，可按需修改）
    image_files = [f for f in os.listdir(folder1) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def rename_images(index, file_name):
        """
        重命名两个文件夹中对应图片的函数
        :param index: 图片的编号
        :param file_name: 原始图片文件名
        """
        # 构建新的文件名
        new_name = f"{index + 1}.jpg"
        # 构建 folder1 中图片的原始路径和新路径
        old_path1 = os.path.join(folder1, file_name)
        new_path1 = os.path.join(folder1, new_name)
        # 构建 folder2 中图片的原始路径和新路径
        old_path2 = os.path.join(folder2, file_name)
        new_path2 = os.path.join(folder2, new_name)

        try:
            # 重命名 folder1 中的图片
            os.rename(old_path1, new_path1)
            # 重命名 folder2 中的图片
            os.rename(old_path2, new_path2)
            print(f"成功将 {file_name} 重命名为 {new_name}")
        except Exception as e:
            print(f"重命名 {file_name} 时出错: {e}")

    # 创建线程列表
    threads = []
    # 遍历所有图片文件
    for i, file in enumerate(image_files):
        # 创建线程
        thread = threading.Thread(target=rename_images, args=(i, file))
        threads.append(thread)
        # 启动线程
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    print("所有图片重命名完成。")