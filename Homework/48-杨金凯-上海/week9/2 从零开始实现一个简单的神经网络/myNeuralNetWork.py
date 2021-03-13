
import numpy as np
class NeuralNetWork:
    def __init__(self,inputnodes, hiddennodes, outputnodes, learningrate):
        self.inputnodes = inputnodes
        self.hiddennodes = hiddennodes
        self.outputnodes = outputnodes
        self.learninggrate = learningrate

        # Y = WX+b
        self.Wih = np.random.normal(0.0,pow(self.inputnodes,-0.5),(self.hiddennodes,self.inputnodes))
        self.Who = np.random.normal(0.0,pow(self.hiddennodes,-0.5),(self.outputnodes,self.hiddennodes))
    def activation_function(self,x):
        return 1.0 / (1 + np.exp(-x))
    def train(self,input_list,label_list):
        # 修改数据类型
        input = np.array(input_list, ndmin=2).T
        label = np.array(label_list, ndmin=2).T
        # 正向传播
        hiddoutput_Z = np.dot(self.Wih,input)
        hiddoutput_A = self.activation_function(hiddoutput_Z)
        final_Z = np.dot(self.Who,hiddoutput_A)
        final_A = self.activation_function(final_Z)
        # 误差计算
        # print(self.Who)
        # print(self.Wih)
        output_errors = label - final_A
        hidd_errors = np.dot(self.Who.T,output_errors*final_A*(1-final_A))
        self.Who += self.learninggrate*np.dot(output_errors*final_A*(1-final_A), np.transpose(hiddoutput_A))
        self.Wih += self.learninggrate * np.dot((hidd_errors * hiddoutput_A*(1-hiddoutput_A)), np.transpose(input))
        # print(self.Who)
        # print(self.Wih)
    def predicted(self,input_list):
        # 正向传播
        input = np.array(input_list, ndmin=2).T
        hiddoutput_Z = np.dot(self.Wih, input)
        hiddoutput_A = self.activation_function(hiddoutput_Z)
        final_Z = np.dot(self.Who, hiddoutput_A)
        final_A = self.activation_function(final_Z)
        print(final_A)
        return  final_A
# 设置参数
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
# 创建网络对象
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#读入训练数据
#open函数里的路径根据数据存储的路径来设定
training_data_file = open("dataset/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()  # 按行读取数据存储到列表中
training_data_file.close()
# 训练
#加入epocs,设定网络的训练循环次数
epochs = 300
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(",")
        inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01  # 数据归一化
        # one_hot 标签
        labels = np.zeros(output_nodes) + 0.01
        labels[int(all_values[0])] = 0.99
        # 训练
        n.train(inputs, labels)
# 测试
#读入训练数据
#open函数里的路径根据数据存储的路径来设定
test_data_file = open("dataset/mnist_test.csv",'r')
test_data_list = test_data_file.readlines()  # 按行读取数据存储到列表中
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(",")
    correct_number = int(all_values[0])

    inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01  # 数据归一化
    # 测试
    outputs = n.predicted(inputs)
    label = np.argmax(outputs)
    print("该图片对应的数字为:", correct_number)
    print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)
#计算图片判断的成功率
scores_array = np.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)



# n.predicted([1.0, 0.5, -1.5])
# n.train([1.0, 0.5, -1.5],[1,0,0])






