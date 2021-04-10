from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
from model.AlexNet import AlexNet
import numpy as np
import utils
import cv2
from keras import backend as K
K.set_image_dim_ordering('tf')
'''
set_image_dim_ordering(dim_ordering)是用于设置图像的维度顺序的，有2个可选参数：
（1）‘th’：即Theano模式，会把通道维放在第二个位置上。
（2）‘tf’：即TensorFlow模式，会把通道维放在最后的位置上。
例：100张RGB三通道的16×32（高为16宽为32）彩色图
th模式下的形式是（100, 3, 16, 32）分别是样本维100张图片、通道维3（颜色通道数）、高、宽
tf模式下的形式是（100, 16, 32, 3）
'''
def generate_arrays_from_file(lines,batch_size):
    '''
    获取一个批次大小的数据
    :param lines: 训练数据
    :param batch_size: 一个批次的数据个数
    :return:
    '''
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            # img = cv2.imread(r".\data\image\train" + '/' + name)
            img = cv2.imread(r"./data/train_image" + '/' + name)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # RGB图
            img = img/255 # 归一化
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始
            i = (i+1) % n
        # 处理图像
        X_train = utils.resize_image(X_train,(224,224))
        X_train = X_train.reshape(-1,224,224,3)
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes= 2)   
        yield (X_train, Y_train)


if __name__ == "__main__":
    # 模型保存的位置
    log_dir = "./logs/"

    # 打开数据集的txt
    with open(r"./data/train_txt/dataset.txt","r") as f:
        lines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    # 建立AlexNet模型
    model = AlexNet()
    
    # 保存的方式，3世代保存一次
    checkpoint_period1 = ModelCheckpoint(
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5', # 模型保存地址
                                    monitor='acc',  # 监视器 正确率
                                    save_weights_only=False, # 若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
                                    save_best_only=True, # 当设置为True时，将只保存在验证集上性能最好的模型
                                    period=3 # CheckPoint之间的间隔的epoch数
                                )

    '''
    目的：定义学习率之后，经过一定epoch迭代之后，模型效果不再提升，该学习率可能已经不再适应该模型。
    需要在训练过程中缩小学习率，进而提升模型。
    如何在训练过程中缩小学习率呢？我们可以使用keras中的回调函数ReduceLROnPlateau。
    与EarlyStopping配合使用，会非常方便。
    monitor：监测的值，可以是accuracy，val_loss,val_accuracy
    factor：缩放学习率的值，学习率将以lr = lr*factor的形式被减少
    patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
    mode：‘auto’，‘min’，‘max’之一 默认‘auto’就行 在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。
    epsilon：阈值，用来确定是否进入检测值的“平原区”
    cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
    min_lr：学习率最小值，能缩小到的下限
    '''
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
                            monitor='acc', 
                            factor=0.5, 
                            patience=3, 
                            verbose=1
                        )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
                            monitor='val_loss', # 监视器 设置为验证集的正确率
                            min_delta=0, 
                            patience=10, 
                            verbose=1
                        )

    # 交叉熵
    model.compile(loss = 'categorical_crossentropy', # 损失函数
            optimizer = Adam(lr=1e-3), # 优化器 初始学习率
            metrics = ['accuracy'])  # 度量为  正确率

    # 一次的训练集大小
    batch_size = 128


    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    
    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size), # 一个批次的数据 （训练集）
            steps_per_epoch=max(1, num_train//batch_size), # 训练集数据可分为多少个批次
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),#一个批次的数据 （测试集）
            validation_steps=max(1, num_val//batch_size), # 测试集数据可分为多少个批次
            epochs=50,# 迭代50代
            initial_epoch=0, # 第一代为0代
            callbacks=[checkpoint_period1, reduce_lr]) # 模型保存方式  和 学习率变化函数
    model.save_weights(log_dir+'last1.h5') # 保存模型到指定位置

