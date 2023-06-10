import os
import shutil
import random


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)

def split_data(cla, src_path, split_rate, dir1, dir2, tag:str = 'copy'):
    """
    按 CustomDataset 格式把数据随机抽样到 dir1、dir2 中
    :param cla: 划分的类别
    :param src_path: 要划分目录的根目录
    :param split_rate: 划分进 dir1 中的比例
    :param dir1:
    :param dir2:
    :param tag: 文件操作意图
    """

    cla_path = src_path + '/' + cla + '/'  # 某一类别的子目录
    images = os.listdir(cla_path)  # images 列表存储了该目录下所有图像的名称
    num = len(images)
    dir1_index = random.sample(images, k=int(num * split_rate))  # 从images列表中随机抽取 k 个图像名称
    for index, image in enumerate(images):
        # dir1_index 中保存 dir1 的图像名称
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar

        if image in dir1_index:
            image_path = cla_path + image
            new_path = dir1 + cla
            print(image_path, new_path)
            if src_path != dir1:
                if tag == 'copy':
                    shutil.copy(image_path, new_path)  # 将选中的图像复制到新路径
                elif tag == 'move':
                    shutil.move(image_path, new_path)
        # 其余的图像保存在 dir2 中
        else:
            image_path = cla_path + image
            new_path = dir2 + cla
            print(image_path, new_path)
            if src_path != dir2:
                if tag == 'copy':
                    shutil.copy(image_path, new_path)  # 将选中的图像复制到新路径
                elif tag == 'move':
                    shutil.move(image_path, new_path)
    return num


if __name__ == '__main__':
    src_path = r"../../mmpretrain/data/fruit"
    dst_path = r"../../mmpretrain/data/fruit-30"

    # 划分比例，训练集 : 验证集 : 测试集 = 8 : 2
    train_rate, val_rate = 0.8, 0.2

    data_class = [cla for cla in os.listdir(src_path)]

    train_path = dst_path + '/train/'
    val_path = dst_path + '/val/'

    # 创建 训练集train 文件夹，并由类名在其目录下创建子目录
    mkfile(dst_path)
    for cla in data_class:
        mkfile(train_path + cla)

    # 创建 验证集val 文件夹，并由类名在其目录下创建子目录
    mkfile(dst_path)
    for cla in data_class:
        mkfile(val_path + cla)


    # 遍历所有类别的全部图像并按比例分成训练集和验证集
    for cla in data_class:
        split_data(cla, src_path, train_rate, train_path, val_path)
        print()
    print("processing done!")

