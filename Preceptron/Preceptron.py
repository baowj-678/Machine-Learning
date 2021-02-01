import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

eta = 0.00001
xs = tf.placeholder(tf.float32, [2, 1])
ys = tf.placeholder(tf.float32, [1])

w = tf.Variable(tf.random_normal([1, 2]), tf.float32)
b = tf.Variable(0.0, tf.float32)


s = ys * (tf.matmul(w, xs) + b)
s = tf.reshape(s, [])
w = tf.cond(s < 0, lambda:tf.add(w, tf.transpose(eta * ys * xs)), lambda:w)
b = tf.cond(s < 0, lambda:tf.add(b, eta * ys), lambda:b)

train = ys * tf.matmul(w, xs) + b

init=tf.global_variables_initializer()

with tf.Session() as sess:
    data = np.array([])
    num = 200
    for i in range(num):
        p = []
        x = np.random.uniform(0, 6)
        y = np.random.uniform(-5, 2)
        if y < ((7 - x * x) / 6):
            p.append(x)
            p.append(y)
            p.append(-1)
        elif y > ((7 - x * x) / 6):
            p.append(x)
            p.append(y)
            p.append(1)
        data = np.append(data, p)
    data = data.reshape([-1, 3])
    print(data.shape)
    data_index = data[:, 2]
    data_plus = data[data_index == 1]
    data_minus = data[data_index == -1]
    plt.scatter(data_plus[:, 0],data_plus[:,1])
    plt.scatter(data_minus[:, 0],data_minus[:,1])

    sess.run(init)
    for i in range(1000):
        for j in range(200):
            x_data = data[j, :2][:, np.newaxis]
            y_data = np.reshape(data[j, 2], [1])
            sess.run(train, feed_dict={xs:x_data, ys:y_data})
    w = sess.run(w, feed_dict={xs:data[1, :2][:, np.newaxis], ys:np.reshape(data[1, 2], [1])})
    b = sess.run(b, feed_dict={xs:data[1, :2][:, np.newaxis], ys:np.reshape(data[1, 2], [1])})
    x = np.linspace(np.min(data, axis=0)[0], np.max(data, 0)[0], 50)
    print(w, b)
    y = (-b - w[0, 0] * x)/w[0, 1]
    plt.plot(x,y)
    plt.show()
