import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt 
learning_rate = 0.03
N = 200

y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[N, 1])
x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2, N])

alpha = tf.Variable(tf.random.uniform(shape=[N, 1], dtype=tf.float32))
b = tf.Variable(initial_value=0.0, dtype=tf.float32)


value = tf.multiply(tf.multiply(alpha, y), tf.transpose(x))
value = tf.math.add(tf.math.reduce_sum(tf.matmul(value, x),axis=1), b)
value = tf.multiply(y, tf.expand_dims(value, axis=-1))
judge = tf.nn.relu(tf.sign(value))

alpha = tf.add(alpha, tf.multiply(alpha, learning_rate))
b = tf.add(b, tf.multiply(tf.math.reduce_sum(tf.multiply(judge, y),axis=0), learning_rate))
w = tf.math.reduce_sum(tf.multiply(tf.multiply(alpha, y), tf.transpose(x)),axis=0)

print(b)
print(alpha)
print(value)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    data = np.array([])
    num = 200
    for i in range(num):
        p = []
        x_data = np.random.uniform(0, 6)
        y_data = np.random.uniform(-5, 2)
        if y_data < ((7 - x_data * x_data) / 6):
            p.append(x_data)
            p.append(y_data)
            p.append(-1)
        elif y_data > ((7 - x_data * x_data) / 6):
            p.append(x_data)
            p.append(y_data)
            p.append(1)
        data = np.append(data, p)
    data = data.reshape([-1, 3])
    print(data.shape)
    data_index = data[:, 2]
    data_plus = data[data_index == 1]
    data_minus = data[data_index == -1]
    plt.scatter(data_plus[:, 0],data_plus[:,1])
    plt.scatter(data_minus[:, 0],data_minus[:,1])
    print(np.transpose(data[:, :2]).shape)
    print(data[:, 2][:, np.newaxis].shape)
    for i in range(100000):
            sess.run(b, feed_dict={x:np.transpose(data[:, :2]), y:data[:, 2][:, np.newaxis]})
    
    w = sess.run(w, feed_dict={x:np.transpose(data[:, :2]), y:data[:, 2][:, np.newaxis]})
    b = sess.run(b, feed_dict={x:np.transpose(data[:, :2]), y:data[:, 2][:, np.newaxis]})
    x = np.linspace(np.min(data, axis=0)[0], np.max(data, 0)[0], 50)
    print(w, b)
    y = (-b - w[0] * x)/w[1]
    plt.plot(x,y)
    plt.show()