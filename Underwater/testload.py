import os
import time

# 定义要检查的硬盘或目录
root_directory = 'C:\\'  # 可以根据需要修改为其他硬盘或目录

# 定义要筛选的代码文件扩展名
code_extensions = ['.py', '.java', '.cpp', '.c', '.js', '.html', '.css']

# 存储代码文件及其存储环境信息的列表
code_files_info = []

# 遍历指定目录及其子目录
for root, dirs, files in os.walk(root_directory):
    for file in files:
        # 获取文件的扩展名
        file_extension = os.path.splitext(file)[1].lower()
        # 检查文件扩展名是否在代码文件扩展名列表中
        if file_extension in code_extensions:
            # 获取文件的完整路径
            file_path = os.path.join(root, file)
            # 获取文件的大小
            file_size = os.path.getsize(file_path)
            # 获取文件的创建时间
            file_creation_time = time.ctime(os.path.getctime(file_path))
            # 获取文件的修改时间
            file_modification_time = time.ctime(os.path.getmtime(file_path))
            # 将文件信息添加到列表中
            code_files_info.append({
                'file_path': file_path,
                'file_size': file_size,
                'file_creation_time': file_creation_time,
                'file_modification_time': file_modification_time
            })

# 打印筛选出的代码文件及其存储环境信息
for info in code_files_info:
    print(f"File Path: {info['file_path']}")
    print(f"File Size: {info['file_size']} bytes")
    print(f"File Creation Time: {info['file_creation_time']}")
    print(f"File Modification Time: {info['file_modification_time']}")
    print("-" * 50)