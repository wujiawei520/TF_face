from sklearn.model_selection import train_test_split
import cv2
import os
import sys
import random
import numpy as np

IMAGE_SIZE = 64

# 返回图片列表
def getfilesinpath(filedir):
   
    for (path, dirnames, filenames) in os.walk(filedir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                yield os.path.join(path, filename)
        for diritem in dirnames:
            getfilesinpath(os.path.join(path, diritem))

# 读取images, labels数据
def readimage(pairpathlabel):

    images = []
    labels = []
    for filepath, label in pairpathlabel:
        for fileitem in getfilesinpath(filepath):
            image = cv2.imread(fileitem)
            image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)


# onehot编码
def onehot(numlist):
    b = np.zeros([len(numlist), max(numlist) + 1])
    b[np.arange(len(numlist)), numlist] = 1
    return b.tolist()

# 获取label
def getUserLabel(filedir):
    dictdir = dict([[name, '%s/%s' % (filedir, name)]\
        for name in os.listdir(filedir) if os.path.isdir(os.path.join(filedir, name))])
    dirnamelist, dirpathlist = dictdir.keys(), dictdir.values()
    indexlist = list(range(len(dirnamelist)))
    return list(zip(dirpathlist,
                    onehot(indexlist))), dict(zip(indexlist, dirnamelist))

#按照指定图像大小调整尺寸
def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)

    #获取图像尺寸
    h, w, _ = image.shape

    #对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)

    #计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top

    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass

    #RGB颜色
    BLACK = [0, 0, 0]

    #给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image,
                                  top,
                                  bottom,
                                  left,
                                  right,
                                  cv2.BORDER_CONSTANT,
                                  value=BLACK)

    #调整图像大小并返回
    return cv2.resize(constant, (height, width))


#从指定路径读取训练数据
def getData(path_name):

    pathlabelpair, indextoname = getUserLabel(path_name)
    images, labels = readimage(pathlabelpair)
    images = images.astype(np.float32) / 255.0
    train_images, test_images, train_labels, test_labels = train_test_split(
            images, labels, test_size=0.3, random_state=random.randint(0, 100))

    return train_images, train_labels, test_images, test_labels

