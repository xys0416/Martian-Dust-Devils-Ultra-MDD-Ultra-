import numpy as np
import cv2
import os
import random, math

class ReadData(object):
    def __init__(self, path, ratio=[0.95, 0.05], size=(256, 256)):
        self.path = path
        self.size = size

        for i, j, k in os.walk(path + "True//"):  # 获取文件夹中的所有文件名称
            catalogList = k  # k是个列表，包含所有该目录下的文件名
        tList = catalogList
        tList = [path + "True//" + tList[ind] for ind in range(len(tList))]
        t_label = np.ones(len(tList))

        for i, j, k in os.walk(path + "False//"):  # 获取文件夹中的所有文件名称
            catalogList = k  # k是个列表，包含所有该目录下的文件名
        fList = catalogList
        fList = [path + "False//" + fList[ind] for ind in range(len(fList))]
        f_label = np.zeros(len(fList))
        tList.extend(fList)
        image_list = tList

        label_list = np.concatenate([t_label, f_label], axis=-1)
        index = list(range(len(label_list)))
        random.shuffle(index)
        image_list = [image_list[ind] for ind in index]
        label_list = [label_list[ind] for ind in index]

        self.train_image_path = image_list[: int(len(index)*ratio[0])]
        self.train_label = label_list[: int(len(index)*ratio[0])]
        self.test_image_path = image_list[int(len(index)*ratio[0]):]
        self.test_label = label_list[int(len(index)*ratio[0]):]

    def image_flip(self, img):
        list_index = [1, 2, 3, 4]
        random.shuffle(list_index)
        if list_index[0] == 1:
            dst = img
        if list_index[0] == 2:
            dst = cv2.flip(img, 0)
        if list_index[0] == 3:
            dst = cv2.flip(img, 1)
        if list_index[0] == 4:
            dst = cv2.flip(img, -1)
        return dst

    def image_rotate(self, img):
        height, width = img.shape[:2]

        degree = random.uniform(-90, 90)
        # 旋转后的尺寸
        heightNew = int(width * math.fabs(math.sin(math.radians(degree))) + height * math.fabs(math.cos(math.radians(degree))))
        widthNew = int(height * math.fabs(math.sin(math.radians(degree))) + width * math.fabs(math.cos(math.radians(degree))))

        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

        matRotation[0, 2] += (widthNew - width) / 2  # 重点在这步，目前不懂为什么加这步
        matRotation[1, 2] += (heightNew - height) / 2  # 重点在这步

        imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))

        return imgRotation

    def change_shape(self, img):
        ratio1 = random.uniform(1, 2)
        ratio2 = random.uniform(1, 2)
        dst = cv2.resize(img, (0, 0), fx=ratio1, fy=ratio2)
        return dst

    def image_resize(self, img):
        ih, iw = img.shape[:2]
        w, h = self.size

        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dst = cv2.resize(img, (nw, nh))
        board = np.zeros((h, w, 3), dtype=np.uint8)

        w_start = (w - nw) // 2
        h_start = (h - nh) // 2
        board[h_start:h_start+nh, w_start:w_start+nw] = dst
        return board



    def readTrainData(self, batchsize=16):
        data_num = 0
        yielddata_image = []
        yielddata_label = []
        while True:

            index = list(range(len(self.train_image_path)))
            random.shuffle(index)

            for ind in index:
                image = cv2.imread(self.train_image_path[ind])
                if image is None:
                    continue
                image = self.image_flip(image)
                # image = self.image_rotate(image)
                image = self.change_shape(image)
                image = self.image_resize(image)

                yielddata_image.append(image)
                yielddata_label.append(self.train_label[ind])
                data_num += 1
                if data_num == batchsize:
                    yield [np.array(yielddata_image), np.array(yielddata_label)]
                    data_num = 0
                    yielddata_image = []
                    yielddata_label = []
                # cv2.imshow("%d"%self.train_label[ind], image)
                # key = cv2.waitKey(0)

    def readTestData(self, batchsize=16):
        data_num = 0
        yielddata_image = []
        yielddata_label = []
        while True:

            index = list(range(len(self.test_image_path)))
            random.shuffle(index)

            for ind in index:
                image = cv2.imread(self.test_image_path[ind])
                if image is None:
                    continue
                image = self.image_flip(image)
                # image = self.image_rotate(image)
                image = self.change_shape(image)
                image = self.image_resize(image)

                yielddata_image.append(image)
                yielddata_label.append(self.test_label[ind])
                data_num += 1
                if data_num == batchsize:
                    yield [np.array(yielddata_image), np.array(yielddata_label)]
                    data_num = 0
                    yielddata_image = []
                    yielddata_label = []
                # cv2.imshow("%d"%self.train_label[ind], image)
                # key = cv2.waitKey(0)







# obj = ReadData(".//Dataset//")
# data = obj.readTestData(200)
# for xxx in range(1000):
#     x, y = next(data)
#     print(x.shape, y.shape)