from matplotlib import pyplot as plt
import cv2 as cv
from pathlib import Path
from PIL import Image
from stitching import Stitcher
import numpy as np
from Image_crop import Crop_images_main
from Image_resize import Resize_images_main
from Image_enhance import Enhance_images_main


def plot_image(img, figsize_in_inches=(5, 5)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()


def get_image_paths(img_set):
    return [str(path.relative_to('.')) for path in Path('imgs').rglob(f'{img_set}*')]



if __name__=='__main__':
    # original_images_folder = 'data2'
    # result_iamges_folder = 'imgs'
    # Crop_images_main(input_folder=original_images_folder, output_folder=result_iamges_folder)
    # Resize_images_main(input_folder=result_iamges_folder, output_folder=result_iamges_folder)
    # Enhance_images_main(input_folder=result_iamges_folder, output_folder=result_iamges_folder)


    # weir_imgs = get_image_paths('weir')

    frame_imgs = get_image_paths('frame')

    # 将图片转为统一尺寸再拼接易成功`
    stitcher = Stitcher(try_use_gpu=True, crop=False, confidence_threshold=0.8)
    panorama = stitcher.stitch(frame_imgs)

    image = Image.fromarray(panorama)
    r, g, b = image.split()
    image = Image.merge('RGB', (b, g, r))

    # 显示图像
    image.show()

    # 保存图像
    image.save('panorama_output_pil.jpg')
