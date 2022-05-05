import numpy as np
import math
import copy
import tkinter
from tkinter.messagebox import *


def load_data(filename):
    '''
    导入数据
    input:  
        file_name(string):文件的存储位置
    output: 
        feature_data(mat):特征
        label_data(mat):标签
        n_class(int):类别的个数
    '''
    # 1、获取特征
    f = open(filename)  # 打开文件
    feature_data = []
    label = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        lines = line.strip().split()
        for i in range(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        label.append(int(float(lines[-1])))
        feature_data.append(feature_tmp)
    f.close()  # 关闭文件

    return np.mat(feature_data), label


def load_test_data(filename):
    '''
    导入测试数据
    '''
    f = open(filename)
    feature_data = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        lines = line.strip().split()
        for i in range(len(lines)):
            feature_tmp.append(float(lines[i]))
        feature_data.append(feature_tmp)
    f.close()  # 关闭文件
    return np.mat(feature_data)


def Normalization(data):
    '''
    样本数据归一化
        input:
            data(mat):样本特征矩阵
        output:
            Nor_feature(mat):归一化的样本特征矩阵
    '''
    m, n = np.shape(data)
    Nor_feature = copy.deepcopy(data)
    sample_sum = np.sqrt(np.sum(np.square(data), axis=1))
    for i in range(n):
        Nor_feature[:, i] = Nor_feature[:, i] / sample_sum

    return Nor_feature


def distance(X, y):
    '''
    计算两个样本之间的距离
    '''
    return np.sum(np.square(X - y), axis=1)


def distance_mat(Nor_trainX, Nor_testX):
    '''
    计算待测试样本与所有训练样本的欧式距离
        input:
            Nor_trainX(mat):归一化的训练样本
            Nor_testX(mat):归一化的测试样本
        output:
            Euclidean_D(mat):测试样本与训练样本的距离矩阵
    '''
    m, n = np.shape(Nor_trainX)  # 查看矩阵或数组的维度
    p = np.shape(Nor_testX)[0]
    Euclidean_D = np.mat(np.zeros((p, m)))
    for i in range(p):
        for j in range(m):
            Euclidean_D[i, j] = distance(
                Nor_testX[i, :], Nor_trainX[j, :])[0, 0]
            # print(Euclidean_D[i, j])
    return Euclidean_D


def Gauss(Euclidean_D, sigma):
    '''
    测试样本与训练样本的距离矩阵对应的Gauss矩阵
        input:
            Euclidean_D(mat):测试样本与训练样本的距离矩阵
            sigma(float):Gauss函数的标准差
        output:
            Gauss(mat):Gauss矩阵
    '''
    m, n = np.shape(Euclidean_D)
    Gauss = np.mat(np.zeros((m, n)))
    for i in range(m):
        for j in range(n):
            Gauss[i, j] = math.exp(- Euclidean_D[i, j] / (2 * (sigma ** 2)))
    return Gauss


def Prob_mat(Gauss_mat, labelX):
    '''
    测试样本属于各类的概率和矩阵
        input:
            Gauss_mat(mat):Gauss矩阵
            labelX(list):训练样本的标签矩阵
        output:
            Prob_mat(mat):测试样本属于各类的概率矩阵
            label_class(list):类别种类列表
    '''
    # 找出所有的标签类别
    label_class = []
    for i in range(len(labelX)):
        if labelX[i] not in label_class:
            label_class.append(labelX[i])

    n_class = len(label_class)
    # 求概率和矩阵
    p, m = np.shape(Gauss_mat)
    Prob = np.mat(np.zeros((p, n_class)))
    for i in range(p):
        for j in range(m):
            for s in range(n_class):
                if labelX[j] == label_class[s]:
                    Prob[i, s] += Gauss_mat[i, j]
    Prob_mat = copy.deepcopy(Prob)
    Prob_mat = Prob_mat / np.sum(Prob, axis=1)
    return Prob_mat, label_class


def class_results(Prob, label_class):
    '''
    分类结果
        input:
            Prob(mat):测试样本属于各类的概率矩阵
            label_class(list):类别种类列表
        output:
            results(list):测试样本分类结果
    '''
    arg_prob = np.argmax(Prob, axis=1)
    results = []
    for i in range(len(arg_prob)):
        results.append(label_class[arg_prob[i, 0]])
    return results


def judge(predict_results):
    flag = 0
    for i in range(len(predict_results)):
        if predict_results[i] == 2:
            flag = 1  # 有污点
    for i in range(len(predict_results)):
        if predict_results[i] == 1:
            if flag == 1:
                return 0  # 有污点和裂痕
            return 1  # 无污点有裂痕
    if flag == 1:
        return 2  # 有污点无裂痕
    return 3  # 无污点无裂痕


def main():
    # 1. 导入数据
    trainX, labelX = load_data(r"src\data\data5.5.1.txt")
    testX = load_test_data("Test_Data.txt")
    # 2、样本数据归一化
    Nor_trainX = Normalization(trainX[:, :])
    Nor_testX = Normalization(testX[:, :])
    # 3、计算Gauss矩阵
    Euclidean_D = distance_mat(Nor_trainX, Nor_testX)
    Gauss_mat = Gauss(Euclidean_D, 0.001)
    Prob, label_class = Prob_mat(Gauss_mat, labelX)
    # 4、求测试样本的分类
    predict_results = class_results(Prob, label_class)
    print(predict_results)
    Judge = judge(predict_results)
    print(Judge)
    if(Judge == 0):
        resultWD = showinfo('检测结果', '这是一颗坏蛋，有污点')
    elif Judge == 1:
        resultWD = showwarning('检测结果', '这是一颗坏蛋,无污点')
    elif Judge == 2:
        resultWD = showinfo('检测结果', '这是一颗好蛋，有污点')
    else:
        resultWD = showinfo('检测结果', '这是一颗好蛋，无污点')
    file = open("Test_Data.txt", 'w').close()


def connect():
    print("HelloWorld!")


main()
