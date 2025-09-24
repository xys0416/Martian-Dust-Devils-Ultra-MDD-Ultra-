import tensorflow as tf
import cv2
import numpy as np
import os
from DeepLabV3APSS import DeepLabV3ASPP
from ReadDataSet import ReadDataset


image_save_dir = "./test_output_images/"  
if not os.path.exists(image_save_dir):  
    os.makedirs(image_save_dir) 
    
label_color = {"head": [255, 0, 255], "shadow": [0, 255, 0]}
model = DeepLabV3ASPP()
model.build(input_shape=(None, 256, 256, 3))
model.summary()
model.load_weights(".//new_beight//DeepLabV3ASPP23000_92.775.ckpt")
opt = tf.optimizers.Adam(learning_rate=1e-4)
datamodel = ReadDataset()

train_data = datamodel.readTrainData(16)
test_data = datamodel.readTestData(8)

epochs = 100000
batch = 1000

for ind_epo in range(epochs):
    x, y = next(train_data)
    x = tf.convert_to_tensor(x, dtype=tf.float32)/255
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    with tf.GradientTape() as tape:
        out = model(x)
        loss = tf.reduce_sum(tf.losses.categorical_crossentropy(y, out))/1
    grads = tape.gradient(loss,model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    jdt = int(100 * (ind_epo % batch) / batch)
    print("\r", end="")
    print("Train Step: {}%: ".format(jdt), "▋" * (jdt // 2), end="")

    if ind_epo % batch == 0:

        x_t_org, y_t, y_t_org = next(test_data)
        x_t = tf.convert_to_tensor(x_t_org, dtype=tf.float32) / 255
        out_t = model(x_t)
        out_t = tf.argmax(out_t, axis=-1).numpy()
        show_boart_background = np.zeros((out_t.shape[0], out_t.shape[1], out_t.shape[2], 3), dtype=np.uint8)
        for ind, type_object in enumerate(label_color.keys()):
            show_boart_background[out_t == ind+1] = label_color[type_object]

        b, h, w, c = x_t_org.shape
        x_t_org = np.transpose(x_t_org, [1, 0, 2, 3])
        x_t_org = np.reshape(x_t_org, [h, b*w, c])

        y_t_org = np.transpose(y_t_org, [1, 0, 2, 3])
        y_t_org = np.reshape(y_t_org, [h, b * w, c])

        show_boart_background = np.transpose(show_boart_background, [1, 0, 2, 3])
        show_boart_background = np.reshape(show_boart_background, [h, b * w, c])
        save_board = np.concatenate([x_t_org, y_t_org, show_boart_background], axis=0)

        imu_target = tf.argmax(y_t, axis=-1).numpy()
        imu = (imu_target == out_t)
        imu.astype(dtype=np.float32)
        imu = np.sum(imu)/(b*h*w)

        filename = f"test_output_epoch_{ind_epo}_loss_{float(loss):.3f}_iou_{imu*100:.3f}.png"
        filepath = os.path.join(image_save_dir, filename)
        cv2.imwrite(filepath, save_board)
        
        print("epochs:", ind_epo, "   loss:", float(loss), "   IOU:%.3f"%(imu*100), "%")
        model.save_weights(".//new_beight//DeepLabV3ASPP%d_%.3f.ckpt"%(ind_epo, imu*100))
        print("\r", end="")
