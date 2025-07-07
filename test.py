from PIL import  Image
import  numpy as np
from matplotlib import pyplot as plt

from model.utils.image import greyscale

if __name__ == '__main__':

    image = np.array(Image.open("data/small/img.png"))
    # 显示原图和灰度图
    plt.rcParams["font.family"] = ["SimHei"]
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("原始 RGB 图像")
    plt.axis("off")
    print(image.shape)
    gray_image = greyscale(image)
    plt.subplot(1, 2, 2)
    plt.imshow(gray_image, cmap="gray")  # 使用 squeeze() 移除单通道维度
    plt.title("灰度处理后的图像")
    plt.axis("off")
    plt.show()