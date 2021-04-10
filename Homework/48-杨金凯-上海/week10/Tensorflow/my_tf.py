import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# [1] 造数据 200个 y = x*x
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data)+noise
print(x_data.shape)
print(y_data.shape)
# [2] 创建占位符
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

# [3] 创建神经网络 1-10-1
Weights_L1 = tf.Variable(tf.random.normal([1,10]))  # 权重变量
Biases_L1 = tf.Variable(tf.zeros([1,10]))           # 偏置值变量
wx_plus_b_L1 = tf.matmul(x,Weights_L1)+Biases_L1      # XW+B
L1 = tf.nn.tanh(wx_plus_b_L1)                         # 激活

Weights_L2 = tf.Variable(tf.random.normal([10,1]))  # 权重变量
Biases_L2 = tf.Variable(tf.zeros([1,1]))           # 偏置值变量
wx_plus_b_L2 = tf.matmul(L1,Weights_L2)+Biases_L2      # XW+B
prediction = tf.nn.tanh(wx_plus_b_L2)                         # 激活

# [4] 定义均方误差损失
loss = tf.reduce_mean(tf.square(y-prediction))

# [5] 定义反向传播
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# [6] 定义会话
with tf.Session() as sess:
    # 初始化全局变量
    sess.run(tf.global_variables_initializer())
    # 循环训练次数
    for i in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    prediction_value = sess.run(prediction,feed_dict={x:x_data})

    plt.plot(x_data,prediction_value,'r-',lw=5)
    plt.scatter(x_data, y_data)
    plt.show()





