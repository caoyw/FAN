
import pandas as pd
import matplotlib.pyplot as plt



if __name__=='__main__':
    data15 = pd.read_csv("../data/jb_train_set.csv")
    data21 = pd.read_csv("../data/jb_test_set.csv")

    # data15 = pd.read_csv("../data/ph_train_set.csv")

    # 取特征和标签
    X_train, y_train = data15.ix[:, 0:-1], data15.ix[:, -1]
    X_test, y_test = data21.ix[:, 0:-1], data21.ix[:, -1]

    # selected_features = ['pitch1_angle']
    #
    # selected_features1 = ['pitch1_angle' ]


    selected_features = ['pitch2_moto_tmp']

    selected_features1 = ['pitch3_moto_tmp']

    X_train1 = X_train[selected_features]
    X_train2 = X_train[selected_features1]
    print(X_train1)
    print(X_train2)
    # plt.plot(X_train1)
    # plt.plot(X_train2)
    # plt.title('test')
    # plt.xlabel('Accuracy')
    # plt.xlabel('Epoch')









    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # matplotlib画图中中文显示会有问题，需要这两行设置默认字体

    plt.xlabel('pitch1_moto_tmp')
    plt.ylabel('pitch2_moto_tmp')

    colors1 = '#00CED1'  # 点的颜色
    area = np.pi * 2 ** 2  # 点面积
    # 画散点图
    plt.scatter(X_train1, X_train2, s=area, c=colors1, alpha=0.4,)

    plt.legend()
    plt.show()


