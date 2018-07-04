import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import linear_model
import random as rd

train_dataDir = '../face/v3/train'
test_dataDir = '../face/v3/test'
all_dataDir = '../face/v3/all'
trans_dataDir = '../face/v3/other'

input_num, n, fw = 231, 8, 10


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(n, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


def get_pca():
    x, _, fwhr = getPoints(all_dataDir)
    pca = PCA(n_components=n)
    pca.fit(x)
    return pca, fwhr.mean(), fwhr.std()


def compute_loss(y1, y2):
    num = 0.0
    for i in range(len(y1)):
        num += (y1[i] - y2[i]) ** 2
    return num / len(y1)


def getPoints(path):
    x_list = []
    y_list = []
    fwhr_list = []

    file_list = os.listdir(path)
    for file_name in file_list:
        with open(os.path.join(path, file_name), 'r') as f:
            points = f.read()
        point_list = points.split(',')
        x = []
        for point in point_list:
            cor_list = point.split(' ')
            p = [float(cor) for cor in cor_list]
            x.append(p)

        x_list.append(x)
        # x_list.append(x[:75])
        y_list.append(float(file_name.split('_')[2][:-4]))

        width = abs(x[1][0] - x[13][0])
        height = abs((x[28][1] + x[32][1]) / 2 - x[7][1])
        fwhr_list.append(width / height)

    x_list = np.array(x_list, dtype=np.float32)
    y_list = np.array(y_list, dtype=np.float32)
    fwhr_list = np.array(fwhr_list, dtype=np.float32)

    x_list = x_list.reshape((-1, input_num))
    y_list = y_list.reshape((-1, 1))
    fwhr_list = fwhr_list.reshape((-1, 1))

    return x_list, y_list, fwhr_list


def getPoints_pca(path, need_np=False):
    x_data, y_data, fwhr_data = getPoints(path)

    pca, fwhr_mean, fwhr_std = get_pca()
    # print(sum(pca.explained_variance_ratio_))

    x_data = pca.transform(x_data)
    fwhr_data = (fwhr_data - fwhr_mean) / fwhr_std

    if not need_np:
        x_data = torch.from_numpy(x_data)
        y_data = torch.from_numpy(y_data)

    return x_data, y_data, fwhr_data


def show(predict, ground_true, title=None):
    x = range(predict.shape[0])
    plt.figure()
    plt.plot(x, predict, 'b--o', label='predict')
    plt.plot(x, ground_true, 'r-^', label='ground true')
    plt.legend()  # 展示图例
    plt.xlabel('index')  # 给 x 轴添加标签
    plt.ylabel('expand')  # 给 y 轴添加标签
    if title:
        plt.title(title)
        # plt.savefig(title + '.jpg')
    plt.show()


def train():
    x_train, y_train, _ = getPoints_pca(train_dataDir)
    x_test, y_test, _ = getPoints_pca(test_dataDir)

    criterion = nn.MSELoss()
    model = LinearRegression()
    # if os.path.exists('params.pkl'):
    #     model.load_state_dict(torch.load('params.pkl'))

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # x_train = x_train.to(device)
    # y_train = y_train.to(device)
    # x_test = x_test.to(device)
    # y_test = y_test.to(device)
    # model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    # optimizer = optim.SGD(model.parameters(), lr=1e-4)

    num_epochs = 30000
    for epoch in range(num_epochs):
        inputs = x_train
        target = y_train

        out = model(inputs)
        loss = criterion(out, target)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print('Epoch[{}/{}], loss: {:.6f}'.format(epoch + 1, num_epochs, loss.data[0]))

    torch.save(model.state_dict(), 'params.pkl')

    model.eval()
    predict = model(x_test)
    loss = criterion(predict, y_test)
    print(loss.data[0])

    predict = predict.cpu().data.numpy()
    test = y_test.cpu().data.numpy()

    show(predict, test)


def eval(path):
    if not os.path.exists('params.pkl'):
        return
    x_data, y_data, _ = getPoints_pca(path)

    model = LinearRegression()
    model.load_state_dict(torch.load('params.pkl'))
    model.eval()
    criterion = nn.MSELoss()

    predict = model(x_data)
    loss = criterion(predict, y_data)
    print(loss.data[0])

    predict = predict.data.numpy()
    test = y_data.data.numpy()

    show(predict, test)


def get_pca_components():
    with open('pca_components.txt', 'r') as f:
        s = f.read().split(' ')
        s = filter(None, s)
        s = [float(p) for p in s]
        m = np.array(s)
        return m.reshape((n, input_num))


def get_weights():
    with open('lr_weights.txt', 'r') as f:
        s = f.read().split(' ')
        s = filter(None, s)
        s = [float(p) for p in s]
        m = np.array(s[:n])
        return m.reshape((1, n)), s[n], s[n + 1], s[n + 2]


def get_pca_means():
    with open('pca_means.txt', 'r') as f:
        s = f.read().split(' ')
        s = filter(None, s)
        s = [float(p) for p in s]
        m = np.array(s)
        return m.reshape((-1, input_num))


def LR():
    x_train, y_train, fwhr_train = getPoints_pca(train_dataDir, need_np=True)
    x_test, y_test, fwhr_test = getPoints_pca(test_dataDir, need_np=True)

    predict_all = []

    lr = linear_model.LinearRegression()
    lr.fit(x_train, y_train)
    predict = lr.predict(x_test)
    loss = compute_loss(predict, y_test)
    print('LR loss:%f' % loss)
    show(predict, y_test, title='LR')
    predict_all.append(predict.reshape((1, -1)).tolist()[0])

    lr_fwhr = linear_model.LinearRegression()
    lr_fwhr.fit(x_train, y_train)
    predict = lr_fwhr.predict(x_test)
    predict += fw * fwhr_test
    loss = compute_loss(predict, y_test)
    print('LR_fwhr loss:%f' % loss)
    show(predict, y_test, title='LR_fwhr')
    predict_all.append(predict.reshape((1, -1)).tolist()[0])

    ridge_cv = linear_model.RidgeCV(alphas=[0.1, 0.5, 1.0, 5.0, 10.0])
    ridge_cv.fit(x_train, y_train)
    predict = ridge_cv.predict(x_test)
    loss = compute_loss(predict, y_test)
    print('RidgeCV loss:%f' % loss)
    show(predict, y_test, title='RidgeCV:L2')
    predict_all.append(predict.reshape((1, -1)).tolist()[0])

    lasso_cv = linear_model.LassoCV(alphas=[0.1, 0.5, 1.0, 5.0, 10.0])
    lasso_cv.fit(x_train, y_train)
    predict = lasso_cv.predict(x_test)
    loss = compute_loss(predict, y_test)
    print('LassoCV loss:%.3f' % loss)
    show(predict, y_test, title='LassoCV:L1')
    predict_all.append(predict.reshape((1, -1)).tolist()[0])

    lassolars_cv = linear_model.LassoLarsCV(cv=10)
    lassolars_cv.fit(x_train, y_train)
    predict = lassolars_cv.predict(x_test)
    loss = compute_loss(predict, y_test)
    print('LassoLarsCV loss:%.3f' % loss)
    show(predict, y_test, title='LassoLarsCV:L1')
    predict_all.append(predict.reshape((1, -1)).tolist()[0])

    bayes_ridge = linear_model.BayesianRidge()
    bayes_ridge.fit(x_train, y_train)
    predict = bayes_ridge.predict(x_test)
    loss = compute_loss(predict, y_test)
    print('BayesianRidge loss:%.3f' % loss)
    show(predict, y_test, title='BayesianRidge')
    predict_all.append(predict.reshape((1, -1)).tolist()[0])

    ard = linear_model.ARDRegression()
    ard.fit(x_train, y_train)
    predict = ard.predict(x_test)
    loss = compute_loss(predict, y_test)
    print('ARDRegression loss:%.3f' % loss)
    show(predict, y_test, title='ARDRegression')
    predict_all.append(predict.reshape((1, -1)).tolist()[0])

    mean_predict = np.array(predict_all)
    mean_predict = mean_predict.mean(axis=0)
    show(mean_predict, y_test, title='Mean_predict')


def k_fold(k=5):
    x_train, y_train, _ = getPoints_pca(train_dataDir, need_np=True)
    x_test, y_test, _ = getPoints_pca(test_dataDir, need_np=True)

    total_num = len(x_train)
    d = total_num // k
    model_list = []

    for i in range(k):
        train_indices = []
        for j in range(total_num):
            if j < i * d or j >= min((i + 1) * d, total_num):
                train_indices.append(j)

        x_train_k = x_train[train_indices]
        y_train_k = y_train[train_indices]
        x_test_k = x_train[i * d:min((i + 1) * d, total_num)]
        y_test_k = y_train[i * d:min((i + 1) * d, total_num)]
        lr = linear_model.LinearRegression()
        lr.fit(x_train_k, y_train_k)
        predict = lr.predict(x_test_k)
        loss = compute_loss(predict, y_test_k)
        print('LR %d loss:%.3f' % (i, loss))
        # show(predict, y_test_k)
        model_list.append(lr)

    predict_all = []
    for lr in model_list:
        predict = lr.predict(x_test)
        predict_all.append(predict.reshape((1, -1)).tolist()[0])
    mean_predict = np.array(predict_all)
    mean_predict = mean_predict.mean(axis=0)
    show(mean_predict, y_test, title='k_fold')
    loss = compute_loss(mean_predict, y_test)
    print('LR loss:%f' % loss)


def k_fold_random(k=5):
    x_train, y_train, fwhr_train = getPoints_pca(all_dataDir, need_np=True)
    # x_test, y_test = getPoints_pca(test_dataDir, need_np=True)

    total_num = len(x_train)
    d = total_num // k
    model_list = []

    for i in range(k):
        test_indices = rd.sample(range(total_num), d)
        train_indices = [i for i in range(total_num) if i not in test_indices]

        x_train_k = x_train[train_indices]
        y_train_k = y_train[train_indices]
        x_test_k = x_train[test_indices]
        y_test_k = y_train[test_indices]
        fwhr_test_k = fwhr_train[test_indices]

        lr = linear_model.LinearRegression()
        lr.fit(x_train_k, y_train_k)
        predict = lr.predict(x_test_k)
        predict += fwhr_test_k * fw
        loss = compute_loss(predict, y_test_k)
        print('LR %d loss:%.3f' % (i, loss))
        show(predict, y_test_k)
        model_list.append(lr)

    # predict_all = []
    # for lr in model_list:
    #     predict = lr.predict(x_test)
    #     predict_all.append(predict.reshape((1, -1)).tolist()[0])
    # mean_predict = np.array(predict_all)
    # mean_predict = mean_predict.mean(axis=0)
    # show(mean_predict, y_test, title='k_fold')
    # loss = compute_loss(mean_predict, y_test)
    # print('LR loss:%f' % loss)


def bootstrap():
    x_all, y_all, fwhr_all = getPoints_pca(all_dataDir, need_np=True)
    total_num = len(x_all)

    indices = range(total_num)
    train_indices = set([])
    for i in indices:
        train_indices.add(rd.sample(indices, 1)[0])
    test_indices = [i for i in indices if i not in train_indices]
    train_indices = list(train_indices)

    x_train = x_all[train_indices]
    y_train = y_all[train_indices]
    x_test = x_all[test_indices]
    y_test = y_all[test_indices]
    fwhr_test = fwhr_all[test_indices]

    lr = linear_model.LinearRegression()
    lr.fit(x_train, y_train)
    predict = lr.predict(x_test)
    predict += fwhr_test * fw
    loss = compute_loss(predict, y_test)
    print('LR loss:%.3f' % loss)
    show(predict, y_test)


def local_predict():
    x_data, y_data, fwhr_data = getPoints(test_dataDir)
    components = get_pca_components()
    means = get_pca_means()
    weights, intercept, fwhr_mean, fwhr_std = get_weights()

    x_data -= means
    x_data = np.dot(x_data, components.T)
    predict = np.dot(x_data, weights.T) + intercept

    fwhr_data = (fwhr_data - fwhr_mean) / fwhr_std
    predict += fwhr_data * fw
    loss = compute_loss(predict, y_data)
    print('LR loss:%f' % loss)
    show(predict, y_data, title='local_predict')


def dump_model():
    pca, fwhr_mean, fwhr_std = get_pca()
    with open('pca_components.txt', 'w') as f:
        for row in pca.components_:
            for p in row:
                f.write('%f ' % p)

    with open('pca_means.txt', 'w') as f:
        for p in pca.mean_:
            f.write('%f ' % p)

    x_all, y_all, _ = getPoints_pca(all_dataDir, need_np=True)
    lr = linear_model.LinearRegression()
    lr.fit(x_all, y_all)
    with open('lr_weights.txt', 'w') as f:
        for p in lr.coef_[0]:
            f.write('%f ' % p)
        f.write('%f ' % lr.intercept_)
        f.write('%f %f' % (fwhr_mean, fwhr_std))


def cal_fwhr(path):
    fwhr_list = {}

    _, fwhr_mean, fwhr_std = get_pca()
    file_list = os.listdir(path)
    for file_name in file_list:
        with open(os.path.join(path, file_name), 'r') as f:
            points = f.read()
            point_list = points.split(',')
            x = []
            for point in point_list:
                cor_list = point.split(' ')
                p = [float(cor) for cor in cor_list]
                x.append(p)
            width = abs(x[1][0] - x[13][0])
            height = abs((x[28][1] + x[32][1]) / 2 - x[7][1])
            fwhr = width / height
            fwhr_list[file_name] = fwhr
            # fwhr_list[file_name] = fw * (fwhr - fwhr_mean) / fwhr_std

    print(fwhr_mean, fwhr_std)
    print(fwhr_list)


if __name__ == '__main__':
    # train()
    # eval(train_dataDir)
    # eval(test_dataDir)
    # LR()
    # k_fold()
    k_fold_random()
    # bootstrap()
    # dump_model()
    # local_predict()
    # cal_fwhr('../face/v3/test')
