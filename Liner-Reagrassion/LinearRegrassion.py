import numpy as np                                  # 矩阵计算包
from utils.features import prepare_for_training     # 数据预处理工具类

class LinearRegrassion:
    # 构造函数
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, nomalize_data=True):
        '''在构造函数中调用数据预处理方法:
            data_processed     是预处理后的结果
            features_mean      是main值
            features_deviation 是数据的标准差
        '''
        (data_processed, features_mean, features_deviation) = prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True)

        self.data = data_processed                   # 预处理后的data （矩阵）
        self.labels = labels                         # 标签（也就是真实值）（列向量）
        self.features_mean = features_mean           # 平均值
        self.features_deviation = features_deviation # 标准差
        self.polynomial_degree = polynomial_degree;  
        self.sinusoid_degree = sinusoid_degree;
        self.nomalize_data = nomalize_data;          # 数据预处理
        self.theta = self.data.shape[1];             # 参数 theta 的个数应该和特征的个数一致，即矩阵的列数

    # 定义训练函数，设置迭代次数为500次
    def train(self, alpha, num_iteration=500):
        cost_history = self.gradient_descent(alpha, num_iteration)
        return self.theta,cost_history

    # 迭代模块
    def gradient_descent(self, alpha, num_iteration):
        cost_history=[]                              # 定义一个数组，来记录每次梯度下降后的损失值是多少
        for i in range(num_iteration):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data,self.labels))
        return cost_history        

    # 一次梯度下降：参数是学习率  
    def gradient_step(self, alpha):
        num_examples = self.data.shape[0]                                      # 样本数量
        prediction = LinearRegrassion.hypothesis(self.data,self.theta)         # 算出所有预测值  （列向量）
        delta = prediction-self.labels                                         # 预测值 - 标签值 （列向量）
        theta = self.theta                                                     # 更新前的 theta
        self.theta = theta-alpha*(1/num_examples)*(np.dot(delta.T,self.data)).T# theta列向量 - 此时要更新的列向量(结尾处转置是因为计算结果是一个行向量，而我们需要列向量)

    # 一次计算出训练集中所有样本的预测值，并用装到一个列向量里。(可以参考笔记中对np.dot()的描述)
    @staticmethod    
    def hypothesis(data, theta):
        return np.dot(data,theta)

    # 损失函数
    def cost_function(self, data, labels):
        num_examples = data.shape[0]
        prediction = LinearRegrassion.hypothesis(self.data,self.theta)     # 得到预测值 （列向量）
        delta = prediction-self.labels                                     # 预测值 - 标签值 （列向量）
        cost = (1/2)*np.dot(delta.T,delta)                # 差的平方累加/总个数
        return cost[0][0]                                                  # cost 是一个1×1的矩阵，所以返回 cost[0][0] 是一个数


    '''
    分界线：以上关于训练集，以下关于测试集
    '''            

    # 用测试集做测试
    def get_cost(self,data,labels):                                        
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.nomalize_data)[0] # 测试集数据预处理
        return self.cost_function(data_processed,labels)
    
    # 预测值
    def predict(self,data):
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.nomalize_data)[0] # 测试集数据预处理
        predictions = self.hypothesis(data_processed,self.theta)                                                                  # 