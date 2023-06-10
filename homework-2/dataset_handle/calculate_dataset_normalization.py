import tqdm
import numpy as np
import cv2
import glob

def get_mean_std(image_path_list):
    # 打印出所有图片的数量
    print('Total images size:', len(image_path_list))
    # 结果向量的初始化,三个维度，和图像一样
    max_val, min_val = np.zeros(3), np.ones(3) * 255
    mean, std = np.zeros(3), np.zeros(3)
    # 利用tqdm模块，可以加载进度条
    for image_path in tqdm.tqdm(image_path_list):  # tqdm用于加载进度条
        # 读取TRAIN中的每一张图片
        image = cv2.imread(image_path)
        # 分别处理三通道
        for c in range(3):
            # 计算每个通道的均值和方差
            mean[c] += image[:, :, c].mean()
            std[c] += image[:, :, c].std()
            max_val[c] = max(max_val[c], image[:, :, c].max())
            min_val[c] = min(min_val[c], image[:, :, c].min())

    # 所有图像的均值和方差
    mean /= len(image_path_list)
    std /= len(image_path_list)
    # 归一化，将值滑到0-1之间
    mean /= max_val - min_val
    std /= max_val - min_val
    # print(max_val - min_val)
    return mean, std


def main():
    # 列表加载储存所有的图片的路径
    image_path_list = []
    # TRAIN中所有的文件
    image_path_list.extend(glob.glob(
        r'../../mmpretrain/data/fruit-30/train/*/**'))  # glob.glob用来返回改路径下所有符合格式要求的所有文件
    # 获得图像的均值和方差
    mean, std = get_mean_std(image_path_list)
    print('mean:', mean)
    print('std:', std)


if __name__ == '__main__':
    main()
