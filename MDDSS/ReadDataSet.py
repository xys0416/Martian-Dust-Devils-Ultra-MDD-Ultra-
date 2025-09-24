import os, random, cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

label_color = {"head": [255, 0, 255], "shadow": [0, 255, 0]}
SavePath_Image = ".//DataSet//Image//"
SavePath_Mask = ".//DataSet//Mask//"


class ReadDataset(object):

    def __init__(self, imgpath=".//DataSet//Image//", maskpath=".//DataSet//Mask//", ratio=[0.9, 0.1], size=(256, 256)):
        self.imgpath = imgpath
        self.maskpath = maskpath
        self.size = size
        for i, j, k in os.walk(SavePath_Image): # 获取文件夹中的所有文件名称
            catalogList = k # k是个列表，包含所有该目录下的文件名
        self.catalogList = catalogList
        sequential = list(range(len(catalogList)))
        random.shuffle(sequential)
        self.traincat = sequential[0:int(ratio[0] * len(catalogList))]
        self.testcat = sequential[int(ratio[0] * len(catalogList)):int((ratio[0] + ratio[1]) * len(catalogList))]

        self.namelist = list(label_color.keys())
        self.colorlist = [label_color[ind] for ind in self.namelist]

    def readTrainData(self, batchsize=16):
        data_num = 0
        yielddata_image = []
        yielddata_label = []

        while True:
            for train_num in self.traincat:
                obj_name = self.catalogList[train_num]
                img_path = self.imgpath + obj_name
                maskpath = self.maskpath + obj_name
                image = cv2.imread(img_path)
                image = cv2.resize(image, self.size)
                label = cv2.imread(maskpath)
                label = cv2.resize(label, self.size)
                yielddata_image.append(image)
                yielddata_label.append(self.anaColor(label))
                # plt.imshow(self.anaColor(label))
                # plt.show()
                data_num += 1

                if data_num == batchsize:
                    yield [np.array(yielddata_image), np.array(yielddata_label)]
                    data_num = 0
                    yielddata_image = []
                    yielddata_label = []

    def readTestData(self, batchsize=16):
        data_num = 0
        yielddata_image = []
        yielddata_label = []
        yielddata_org_label = []
        while True:
            for test_num in self.testcat:
                obj_name = self.catalogList[test_num]
                img_path = self.imgpath + obj_name
                maskpath = self.maskpath + obj_name
                image = cv2.imread(img_path)
                image = cv2.resize(image, self.size)
                label = cv2.imread(maskpath)
                label = cv2.resize(label, self.size)
                yielddata_image.append(image)
                yielddata_label.append(self.anaColor(label))
                yielddata_org_label.append(label)
                # plt.imshow(self.anaColor(label))
                # plt.show()
                data_num += 1

                if data_num == batchsize:
                    yield [np.array(yielddata_image), np.array(yielddata_label), np.array(yielddata_org_label)]
                    data_num = 0
                    yielddata_image = []
                    yielddata_label = []
                    yielddata_org_label = []

    def anaColor(self, label):
        mb = np.zeros([label.shape[0], label.shape[1]])
        for ind, color in enumerate(self.colorlist):
            mask = (label == np.ones_like(label)*color)
            mask = (mask[..., 0] * mask[..., 1] * mask[..., 2]) * (ind+1)
            mb += mask
        return tf.one_hot(mb, depth=1+len(self.colorlist))

# datamodel = ReadDataset()
# data = datamodel.readTestData(32)
# x, y = next(data)
# test_data = datamodel.readTestData(4)
# for _ in range(10):
#     x, y = next(data)
#     print(x.shape, y.shape)
#     print("__________________________")
#     x, y = next(test_data)
#     print(x.shape, y.shape)
#     print("__________________________")
