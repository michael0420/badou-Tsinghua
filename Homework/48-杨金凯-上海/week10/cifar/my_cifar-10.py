import tensorflow as tf
import Cifar10_data

def variable_with_weight_loss(shape,stddev,w1):
    var=tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        weights_loss=tf.multiply(tf.nn.l2_loss(var),w1,name="weights_loss")
        tf.add_to_collection("losses",weights_loss)
    return var

data_dir="Cifar_data/cifar-10-batches-bin"
max_steps=20000
batch_size=1
num_examples_for_eval=10000
images_train,labels_train=Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=True)
images_test,labels_test=Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=None)
print(images_train.shape)  # (100, 24, 24, 3)
print(labels_train.shape)  # (100,)
# print(images_test.shape)
# print(labels_test.shape)

# [1] 创建占位符
x = tf.placeholder(tf.float32,[batch_size,24,24,3])
y_ = tf.placeholder(tf.int32,[batch_size])

# [2] 创建卷积层1
weight_conv_L1 = variable_with_weight_loss([5,5,3,64],stddev=5e-2,w1=0.0)
bias_L1 = tf.Variable(tf.constant(0.0,shape = [64]))
conv_L1 = tf.nn.conv2d(x,weight_conv_L1,[1,1,1,1],padding = "SAME")
result_L1 = tf.nn.relu(tf.nn.bias_add(conv_L1,bias_L1))
pool_L1 = tf.nn.max_pool(result_L1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")
# [3] 创建卷积层2
weight_conv_L2 = variable_with_weight_loss([5,5,3,64],stddev=5e-2,w1=0.0)
bias_L2 = tf.Variable(tf.constant(0.0,shape = [64]))
conv_L2 = tf.nn.conv2d(x,weight_conv_L2,[1,1,1,1],padding = "SAME")
result_L2 = tf.nn.relu(tf.nn.bias_add(conv_L2,bias_L2))
pool_L2 = tf.nn.max_pool(result_L2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")
# [4] 卷积层过渡全连接层
reshape = tf.reshape(pool_L2,shape=[batch_size,-1])
dim=reshape.get_shape()[1].value  # 获取reshape的第二个维度
# [5] 全连接层1
weight_fc1 = variable_with_weight_loss([dim,384],stddev=0.04,w1=0.04)
bias_fc1 = tf.Variable(tf.constant(0.1,shape=[384]))
fc1 = tf.matmul(reshape,weight_fc1)+bias_fc1
result_fc1 = tf.nn.relu(fc1)
# [6] 全连接层2
weight_fc2 = variable_with_weight_loss([384,192],stddev=0.04,w1=0.04)
bias_fc2 = tf.Variable(tf.constant(0.1,shape=[192]))
fc2 = tf.matmul(result_fc1 ,weight_fc2)+bias_fc2
result_fc2 = tf.nn.relu(fc2)
# [7] 全连接层2
weight_fc3 = variable_with_weight_loss([192,10],stddev=0.04,w1=0.04)
bias_fc3 = tf.Variable(tf.constant(0.1,shape=[10]))
fc3 = tf.matmul(result_fc2,weight_fc3)+bias_fc3
result_fc3 = tf.nn.relu(fc3)
# [8] 计算损失，包括权重参数的正则化损失和交叉熵损失
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result_fc3,labels=tf.cast(y_,tf.int64))
# [9] L2正则化项
weight_l2_regularization_loss = tf.add_n(tf.get_collection("losses"))
# [10] 总损失 = 交叉熵损失的均值+L2正则化
loss = tf.reduce_mean(cross_entropy)+weight_l2_regularization_loss
# [11] 反向传播
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

top_k_op=tf.nn.in_top_k(result_fc3,y_,1)  # result_fc3的前k大的下标是否与y_的前k大的下标相同
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners()
    image_batch, label_batch = sess.run([images_train, labels_train])
    print(image_batch,label_batch)
    # print(sess.run(result_fc3,feed_dict={x:image_batch,y_:label_batch}))




