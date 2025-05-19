from stitching import Stitcher
from PIL import Image
import numpy as np
from datetime import datetime


start_time = datetime.now()
stitcher = Stitcher(detector="sift", confidence_threshold=0.2)
panorama = stitcher.stitch(["screenshots/frame_0000.jpg", "screenshots/frame_0001.jpg",
                            "screenshots/frame_0002.jpg", "screenshots/frame_0003.jpg",
                            "screenshots/frame_0004.jpg", "screenshots/frame_0005.jpg",
                            "screenshots/frame_0006.jpg", "screenshots/frame_0007.jpg"])

middle_time = datetime.now()
print(middle_time - start_time)
# 如果panorama是numpy数组且值范围不在0-255，需要进行转换
if isinstance(panorama, np.ndarray):
    if panorama.dtype != np.uint8:
        if panorama.max() <= 1:
            panorama = (panorama * 255).astype(np.uint8)
        else:
            panorama = panorama.astype(np.uint8)

# 将numpy数组转换为PIL图像
image = Image.fromarray(panorama)
image = image.convert('RGB')


r, g, b = image.split()

image = Image.merge('RGB', (b, g, r))

end_time = datetime.now()

print(end_time - middle_time)

# 显示图像
image.show()

# 保存图像
image.save('panorama_output_pil.jpg')




