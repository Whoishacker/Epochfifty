import numpy as np
import cv2

def calculate_uciqe(img):
    img_BGR = img
    img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
    img_LAB = np.array(img_LAB, dtype=np.float64)
    coe_Metric = [0.4680, 0.2745, 0.2576]

    img_lum = img_LAB[:, :, 0] / 255.0
    img_a = img_LAB[:, :, 1] / 255.0
    img_b = img_LAB[:, :, 2] / 255.0

    # 色度标准差
    chroma = np.sqrt(np.square(img_a) + np.square(img_b))
    sigma_c = np.std(chroma)

    # 亮度对比度
    img_lum = img_lum.flatten()
    sorted_index = np.argsort(img_lum)
    top_index = sorted_index[int(len(img_lum) * 0.99)]
    bottom_index = sorted_index[int(len(img_lum) * 0.01)]
    con_lum = img_lum[top_index] - img_lum[bottom_index]

    # 饱和度均值
    chroma = chroma.flatten()
    sat = np.divide(chroma, img_lum, out=np.zeros_like(chroma, dtype=np.float64), where=img_lum != 0)
    avg_sat = np.mean(sat)

    uciqe = sigma_c * coe_Metric[0] + con_lum * coe_Metric[1] + avg_sat * coe_Metric[2]
    return uciqe

if __name__ == "__main__":
    img_UCIQE = cv2.imread("C:/Users/24312/Desktop/watercompute/output.jpg")
    result_UCIQE = calculate_uciqe(img_UCIQE)
    print(result_UCIQE)