import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


learning_rate = 0.00003
x = tf.compat.v1.placeholder(tf.float32, [2, None])
y = tf.compat.v1.placeholder(tf.float32, [1, None])

w = tf.Variable(tf.random.normal([1, 2]), dtype=tf.float32)
b = tf.Variable(tf.random.normal([1,1]), dtype=tf.float32)

sign = tf.multiply(y, tf.add(tf.matmul(w, x), b))
sign = tf.nn.relu(tf.sign(sign))

delt_w = tf.multiply(sign, y)
delt_w = tf.multiply(delt_w, x)

w = tf.add(w, tf.multiply(tf.math.reduce_sum(delt_w, axis=1), learning_rate))
b = tf.add(b, tf.multiply(tf.math.reduce_sum(sign), learning_rate))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    data = np.array([])
    num = 200
    for i in range(num):
        p = []
        x_ = np.random.uniform(0, 6)
        y_ = np.random.uniform(-5, 2)
        if y_ < ((7 - x_ * x_) / 6):
            p.append(x_)
            p.append(y_)
            p.append(-1)
        elif y_ > ((7 - x_ * x_) / 6):
            p.append(x_)
            p.append(y_)
            p.append(1)
        data = np.append(data, p)
    data = data.reshape([-1, 3])
    print(data.shape)
    data_index = data[:, 2]
    data_plus = data[data_index == 1]
    data_minus = data[data_index == -1]
    plt.scatter(data_plus[:, 0],data_plus[:,1])
    plt.scatter(data_minus[:, 0],data_minus[:,1])
    print(data[:, :2].shape)
    print(data[:,2].shape)
    for i in range(3000):
        for j in range(10):
            sess.run(b, feed_dict={x:np.transpose(data[j*20:j*20 + 20, :2]), y:data[j*20:j*20 + 20:, 2][np.newaxis, :]})
        print(i)
    w = sess.run(w, feed_dict={x:np.transpose(data[:, :2]), y:data[:, 2][np.newaxis, :]})
    b = sess.run(b, feed_dict={x:np.transpose(data[:, :2]), y:data[:, 2][np.newaxis, :]})
    x = np.linspace(np.min(data, axis=0)[0], np.max(data, 0)[0], 50)
    print(w, b)
    y = (-b - w[0, 0] * x)/w[0, 1]
    plt.plot(x, np.reshape(y, [-1, 1]))
    plt.show()