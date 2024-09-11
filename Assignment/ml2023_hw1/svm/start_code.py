import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import trange


def load_text_dataset(filename, positive='joy', negative='sadness'):
    """
    从文件filename读入文本数据集
    """
    data = pd.read_csv(filename)
    is_positive = data.Emotion == positive
    is_negative = data.Emotion == negative
    data = data[is_positive | is_negative]
    X = data.Text  # 输入文本
    y = np.array(data.Emotion == positive) * 2 - 1  # 1: positive, -1: negative
    return X, y


def vectorize(train, test):
    """
    将训练集和验证集中的文本转成向量表示
    Args：
        train - 训练集，大小为 num_instances 的文本数组
        test - 测试集，大小为 num_instances 的文本数组
    Return：
        train_normalized - 向量化的训练集 (num_instances, num_features)
        test_normalized - 向量化的测试集 (num_instances, num_features)
    """
    tfidf = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True)
    train_normalized = tfidf.fit_transform(train).toarray()
    test_normalized = tfidf.transform(test).toarray()
    return train_normalized, test_normalized


def linear_svm_subgrad_descent(X, y, alpha=0.05, lambda_reg=0.0001, num_iter=60000, batch_size=1):
    """
    线性SVM的随机次梯度下降

    参数：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        alpha - 浮点数。梯度下降步长，可自行调整为默认值以外的值或扩展为步长策略
        lambda_reg - 正则化系数，可自行调整为默认值以外的值
        num_iter - 要运行的迭代次数，可自行调整为默认值以外的值
        batch_size - 批大小，可自行调整为默认值以外的值

    返回：
        theta_hist - 参数向量的历史，大小的 2D numpy 数组 (num_iter+1, num_features)
        loss hist - 小批量损失函数的历史，数组大小(num_iter)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    theta_hist[0] = theta = np.zeros(num_features)  # Initialize theta
    loss_hist = np.zeros(num_iter)  # Initialize loss_hist

    # TODO 3.4.1
    for i in range(num_iter):
        # 从训练集中随机抽取 batch_size 个样本
        sample_index = np.random.choice(num_instances, batch_size, replace=True)
        X_train_sample = X[sample_index]
        y_train_sample = y[sample_index]
        # 梯度分段示性向量
        indicator = np.zeros(batch_size)
        indicator[np.where(1-y_train_sample*(X_train_sample@theta_hist[i]) > 0)] = 1
        # 计算损失函数
        loss_hist[i] = (lambda_reg/2*theta_hist[i][0:num_features-1]@theta_hist[i][0:num_features-1] + indicator.T@(1-X_train_sample@theta_hist[i]*y_train_sample))/batch_size
        # 计算梯度
        grad = np.hstack((lambda_reg*theta_hist[i][0:num_features-1],0)) - X_train_sample.T@(indicator*y_train_sample)/batch_size
        # 更新参数
        theta_hist[i+1] = theta_hist[i] - alpha*grad
        print("iter: ", i, "loss: ", loss_hist[i])
    return theta_hist, loss_hist

def linear_svm_subgrad_descent_lambda(X, y, lambda_reg=0.0001, num_iter=60000, batch_size=1):
    """
    线性SVM的随机次梯度下降;在lambda-强凸条件下有理论更快收敛速度的算法
    该函数每次迭代的梯度下降步长已由算法给出，无需自行调整

    参数：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        lambda_reg - 正则化系数，可自行调整为默认值以外的值
        num_iter - 要运行的迭代次数，可自行调整为默认值以外的值
        batch_size - 批大小，可自行调整为默认值以外的值

    返回：
        theta_hist - 参数向量的历史，大小的 2D numpy 数组 (num_iter+1, num_features)
        loss hist - 小批量损失函数的历史，数组大小(num_iter)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    theta_hist[0] = theta = np.zeros(num_features)  # Initialize theta
    loss_hist = np.zeros(num_iter)  # Initialize loss_hist

    # TODO 3.4.3
    for i in range(num_iter):
        # 从训练集中随机抽取 batch_size 个样本
        sample_index = np.random.choice(num_instances, batch_size, replace=True)
        X_train_sample = X[sample_index]
        y_train_sample = y[sample_index]
        # 梯度分段示性向量
        indicator = np.zeros(batch_size)
        indicator[np.where(1-y_train_sample*(X_train_sample@theta_hist[i]) > 0)] = 1
        # 计算损失函数
        loss_hist[i] = (lambda_reg/2*theta_hist[i][0:num_features-1]@theta_hist[i][0:num_features-1] + indicator.T@(1-X_train_sample@theta_hist[i]*y_train_sample))/batch_size
        # 计算梯度
        grad = np.hstack((lambda_reg*theta_hist[i][0:num_features-1],0)) - X_train_sample.T@(indicator*y_train_sample)/batch_size
        # 更新参数
        theta_hist[i+1] = theta_hist[i] - 1/(lambda_reg*(i+1))*grad
        print("iter: ", i, "loss: ", loss_hist[i])
    return theta_hist, loss_hist
    

def kernel_svm_subgrad_descent(X, y, eta=0.1, lambda_reg=1, num_iter=1000, batch_size=1, kernel="linear"):
    """
    Kernel SVM的随机次梯度下降

    参数：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        eta - 浮点数。初始梯度下降步长
        lambda_reg - 正则化系数
        num_iter - 遍历整个训练集的次数（即次数）
        batch_size - 批大小
        kernel - 核函数，可选"linear"或"gussian"

    返回：
        theta_hist - 参数向量的历史，大小的 2D numpy 数组 (num_iter, num_features)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.zeros(num_features)  # Initialize theta
    theta_hist = np.zeros((num_iter+1, num_features))  # Initialize theta_hist
    loss_hist = np.zeros(num_iter+1)  # Initialize loss_hist
    # TODO 3.4.4
    # 核函数确定
    if kernel == "linear":
        def K(x1, x2):
            return x1@x2
        def Phi(x):
            return x
    elif kernel == "gaussian":
        def K(x1, x2):
            return np.exp(-np.linalg.norm(x1-x2)**2/2)
        def Phi(x):
            return np.exp(-x**2/2)
    # 如果有核矩阵则直接读取，否则计算核矩阵并保存
    try:
        K_train = np.load("K_train_"+kernel+".npy")
    except:
        K_train = np.zeros((num_instances, num_instances))
        for i in range(num_instances):
            for j in range(num_instances):
                K_train[i,j] = K(X[i], X[j])
        np.save("K_train_"+kernel+".npy", K_train)
    # 计算Phi(X)
    Phi_X = np.zeros((num_instances, num_features))
    for i in range(num_instances):
        Phi_X[i] = Phi(X[i])
    # 初始化 alpha_hist
    alpha_hist = np.zeros((num_iter+1, num_instances))
    alpha_hist[0] = np.zeros(num_instances)
    theta_hist[0] = np.zeros(num_features)
    # 迭代
    for i in range(num_iter):
        # 从训练集中随机抽取 batch_size 个样本
        sample_index = np.random.choice(num_instances, batch_size, replace=True)
        K_train_sample = K_train[sample_index]
        y_train_sample = y[sample_index]
        # 梯度分段示性向量
        indicator = np.zeros(batch_size)
        indicator[np.where(y_train_sample*(alpha_hist[i]@K_train[sample_index].T) < 1)] = 1
        # 计算次梯度
        subgrad = lambda_reg*K_train@alpha_hist[i] - K_train[sample_index].T@(y_train_sample*indicator)/batch_size
        # 更新 alpha
        alpha_hist[i+1] = alpha_hist[i] - eta*subgrad
        # 更新 theta
        alpha_sum = np.zeros(num_instances)
        
        if(i%100==0):
            print("iter: ", i)
    for j in range(num_iter):
        alpha_sum += alpha_hist[j]
    alpha_avg = alpha_sum/num_iter
    theta_hist[-1] = Phi_X.T@alpha_avg
    return theta_hist




def main():
    # 加载所有数据
    X_train, y_train = load_text_dataset("data_train.csv", "joy", "sadness")
    X_val, y_val = load_text_dataset("data_test.csv")
    print("Training Set Size: {} Validation Set Size: {}".format(len(X_train), len(X_val)))
    print("Training Set Text:", X_train, sep='\n')

    # 将训练集和验证集中的文本转成向量表示
    X_train_vect, X_val_vect = vectorize(X_train, X_val)
    X_train_vect = np.hstack((X_train_vect, np.ones((X_train_vect.shape[0], 1))))  # 增加偏置项
    X_val_vect = np.hstack((X_val_vect, np.ones((X_val_vect.shape[0], 1))))  # 增加偏置项

    # SVM的随机次梯度下降训练
    # TODO
    import matplotlib.pyplot as plt
    num_iter = 10000
    batch_size = 16
    lambda_reg = 0.00025
    theta_hist, loss_hist = linear_svm_subgrad_descent_lambda(X_train_vect, y_train,  lambda_reg=lambda_reg, num_iter=num_iter, batch_size=batch_size)

    # 计算SVM模型在验证集上的准确率，F1-Score以及混淆矩阵
    # TODO
    # 计算正确率(训练集/验证集)
    y_pred = np.sign(X_val_vect@theta_hist[-1])
    val_accuracy = np.sum(y_pred==y_val)/len(y_val)
    print("val_accuracy: ", val_accuracy)

    # 计算F1-Score
    TP = np.sum(np.logical_and(y_pred==1, y_val==1))
    TN = np.sum(np.logical_and(y_pred==-1, y_val==-1))
    FP = np.sum(np.logical_and(y_pred==1, y_val==-1))
    FN = np.sum(np.logical_and(y_pred==-1, y_val==1))
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = 2*precision*recall/(precision+recall)
    print("F1-Score: ", F1)
    # 计算混淆矩阵
    confusion_matrix = np.zeros((2,2))
    confusion_matrix[0,0] = TP
    confusion_matrix[0,1] = FN
    confusion_matrix[1,0] = FP
    confusion_matrix[1,1] = TN
    print("confusion_matrix: ", confusion_matrix)



if __name__ == '__main__':
    main()