import tensorflow as tf
import cv2
import numpy as np


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filters, lay=2, atrous=1, maxpool=True):
        super(ResBlock, self).__init__()
        self.maxpool = maxpool
        self.conv_a = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=1, padding="same", activation="relu")
        self.model = tf.keras.Sequential()
        for lay_i in range(lay):
            self.model.add(tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="same", strides=1, activation="relu", dilation_rate=(atrous, atrous)))
            self.model.add(tf.keras.layers.BatchNormalization())
        if maxpool:
            self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")

    def call(self, inputs):
        out = self.conv_a(inputs)
        out = self.model(out)+out
        if self.maxpool:
            out = self.maxpool(out)
        return out


class DeepLabV3ASPP(tf.keras.Model):

    def __init__(self):

        super(DeepLabV3ASPP, self).__init__()
        self.res_in = tf.keras.layers.Conv2D(64, kernel_size=(5,5), strides=1, padding="same", activation="relu")
        self.res_1 = ResBlock(64)
        self.res_2 = ResBlock(128)
        self.res_3 = ResBlock(256)
        self.res_4 = ResBlock(256, lay=3, atrous=2, maxpool=False)

        self.ASPP_1 = tf.keras.layers.Conv2D(256, kernel_size=(1,1), strides=1, padding="same", activation="relu")
        self.ASPP_2 = tf.keras.layers.Conv2D(256, kernel_size=(3,3), strides=1, padding="same", activation="relu", dilation_rate=(6, 6))
        self.ASPP_3 = tf.keras.layers.Conv2D(256, kernel_size=(3,3), strides=1, padding="same", activation="relu", dilation_rate=(12, 12))
        self.ASPP_4 = tf.keras.layers.Conv2D(256, kernel_size=(5,5), strides=1, padding="same", activation="relu", dilation_rate=(18, 18))
        self.res_in = tf.keras.layers.Conv2D(256, kernel_size=(1,1), strides=1, padding="same", activation="relu")
        self.ASPP_5 = tf.keras.layers.MaxPooling2D(pool_size=(8,8), strides=8, padding="same")

        self.deconv = tf.keras.layers.Conv2D(3, kernel_size=(1,1), strides=1, padding="same", activation="softmax")

    def call(self, inputs):

        out = self.res_4(self.res_3(self.res_2(self.res_1(self.res_in(inputs)))))

        out = tf.concat([self.ASPP_1(out), self.ASPP_2(out), self.ASPP_3(out), self.ASPP_4(out), self.ASPP_5(self.res_in(inputs))], axis=-1)
        out = self.deconv(out)
        out = tf.image.resize(out, (256, 256))
        return out


# model = DeepLabV3ASPP()
# model.build(input_shape=(None, 256, 256, 3))
# model.summary()

