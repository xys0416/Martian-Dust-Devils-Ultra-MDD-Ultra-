import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score
from Resnet import ResNet
from ReadDataset import ReadData

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

def calculate_ap(y_true, y_scores):
    y_scores = np.clip(y_scores, 0, 1)
    ap = average_precision_score(y_true, y_scores)
    return ap

def main():
    batch_size = 32
    epochs = 10000
    initial_learning_rate = 1e-5
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True
    )
    opt = tf.optimizers.Adam(learning_rate=learning_rate_schedule)

    datamodel = ReadData(".//Dataset//")
    data = datamodel.readTrainData(batch_size)
    datatest = datamodel.readTestData(128)
    model = ResNet([16, 16, 16], 2)
    model.load_weights(".//new_beight//Resnet_Epo_5000_128_126.ckpt")

    # 初始化CSV文件，用于保存损失、学习率、AP、准确率、查准率和查全率
    csv_file = 'metrics_history.csv'
    with open(csv_file, 'w') as f:
        f.write('epoch,loss,learning_rate,ap,accuracy,precision,recall\n')

    for epoch in range(epochs):
        x, y = next(data)
        x = tf.convert_to_tensor(x)
        x = tf.cast(x, tf.float32) / 255
        y = tf.convert_to_tensor(y)
        y = tf.cast(y, tf.int32)
        one_hot_labels = tf.one_hot(y, depth=2, dtype=tf.float32)

        with tf.GradientTape() as tape:
            out = model(x)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot_labels, logits=out))

        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 1000 == 0:
            xt, yt = next(datatest)
            xt = tf.convert_to_tensor(xt)
            xt = tf.cast(xt, tf.float32) / 255
            outt = model(xt)
            outt_probs = tf.nn.softmax(outt, axis=-1)[:, 1]
            outt_labels = tf.argmax(outt, axis=-1)
            yt = tf.cast(yt, tf.float32)
            outt_probs = outt_probs.numpy()
            yt = yt.numpy()
            outt_labels = outt_labels.numpy()

            ap = calculate_ap(yt, outt_probs)
            accuracy = np.mean(outt_labels == yt)
            precision = precision_score(yt, outt_labels)
            recall = recall_score(yt, outt_labels)

            # 保存损失、学习率、AP、准确率、查准率和查全率到CSV
            current_lr = opt.learning_rate(opt.iterations).numpy()
            with open(csv_file, 'a') as f:
                f.write(f'{epoch+1},{loss.numpy()},{current_lr},{ap},{accuracy},{precision},{recall}\n')

            # 保存模型权重
            model.save_weights(f".//new_beight//Resnet_Epo_{epoch}_128_{int(ap*100)}.ckpt")
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.numpy()}, Learning Rate: {current_lr}, AP: {ap}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')

if __name__ == '__main__':
    main()
