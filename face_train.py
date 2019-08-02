import os
import cv2
import logging as log
import numpy as np
import tensorflow as tf

SIZE = 64

x_data = tf.placeholder(tf.float32, [None, SIZE, SIZE, 3])
y_data = tf.placeholder(tf.float32, [None, None])

keep_prob = tf.placeholder(tf.float32)


def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)


def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxPool(x):
    return tf.nn.max_pool(x,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')


def dropout(x, keep):
    return tf.nn.dropout(x, keep)


def cnnLayer(classnum):
    # 第一层
    W1 = weightVariable([3, 3, 3, 32])
    b1 = biasVariable([32])

    conv1 = tf.nn.relu(conv2d(x_data, W1) + b1)
    pool1 = maxPool(conv1)
    drop1 = dropout(pool1, keep_prob)

    # 第二层
    W2 = weightVariable([3, 3, 32, 64])
    b2 = biasVariable([64])

    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob)

    # 第三层
    W3 = weightVariable([3, 3, 64, 64])
    b3 = biasVariable([64])

    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob)

    # 全连接层 图片尺寸变化：卷积padding为same方式输出尺寸不变，经三次池化64/2/2/2=8
    Wf = weightVariable([8 * 8 * 64, 512])
    bf = biasVariable([512])

    drop3_flat = tf.reshape(drop3, [-1, 8 * 8 * 64])
    hf = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(hf, keep_prob)

    # 输出层
    Wout = weightVariable([512, classnum])
    bout = weightVariable([classnum])

    out = tf.add(tf.matmul(dropf, Wout), bout)

    return out


def train(train_x, train_y, test_x, test_y, save_path):

    out = cnnLayer(train_y.shape[1])
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_data))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_data, 1)), tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        batch_size = 10
        num_batch = len(train_x) // 10

        for n in range(50):

            r = np.random.permutation(len(train_x))

            train_x = train_x[r, :]
            train_y = train_y[r, :]

            for i in range(num_batch):

                batch_x = train_x[i * batch_size:(i + 1) * batch_size]
                batch_y = train_y[i * batch_size:(i + 1) * batch_size]

                _, loss = sess.run([train_step, cross_entropy],
                                   feed_dict={
                                       x_data: batch_x,
                                       y_data: batch_y,
                                       keep_prob: 0.75
                                   })
                print(n * num_batch + i, loss)

            # 获取训练数据的准确率 
            train_acc = accuracy.eval({
                    x_data: batch_x,
                    y_data: batch_y,
                    keep_prob: 0.75})
            print('step %d, training accuracy %g' % (n * num_batch + i, train_acc))

        # 获取测试数据的准确率
        test_acc = accuracy.eval({
            x_data: test_x,
            y_data: test_y,
            keep_prob: 1.0,
        })
        print('after all, test accuracy is ', test_acc)

        saver.save(sess, save_path)