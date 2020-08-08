import tensorflow as tf

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC as SVM
from sklearn.linear_model import LogisticRegression as LG
from sklearn.ensemble import AdaBoostClassifier  as ADB
from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM,SimpleRNN,Bidirectional


import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from xgboost import plot_importance
from matplotlib import pyplot

import numpy
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score



class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_auc = []


    def on_epoch_end(self, epoch, logs={}):
        val_predict = (numpy.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        _val_auc = roc_auc_score(val_targ, val_predict)
        print("  f1:",_val_f1)
        print("recal:", _val_recall)
        print("pre:",_val_precision)
        print("auc:", _val_auc)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_auc.append(_val_auc)
        return





if __name__=='__main__':










    # Step1 读取数据
    # data15 = pd.read_csv("../data/jb_train_set.csv")
    # data21 = pd.read_csv("../data/jb_test_set.csv")


    # data15 = pd.read_csv("../data/ph_train_set.csv")
    # data21 = pd.read_csv("../data/ph_test_set.csv")

    data15 = pd.read_csv("../data/ph_4_set.csv")
    data21 = pd.read_csv("../data/ph_3_set.csv")

    # 取特征和标签
    X_train, y_train = data15.ix[:, 0:-1], data15.ix[:, -1]
    X_test, y_test = data21.ix[:, 0:-1], data21.ix[:, -1]







    # selected_features = ['wind_speed','generator_speed','power','wind_direction',
    #                      'wind_direction_mean','yaw_position','yaw_speed','pitch1_angle',
    #                      'pitch2_angle','pitch3_angle','pitch1_speed','pitch2_speed','pitch3_speed',
    #                      'pitch1_moto_tmp','pitch2_moto_tmp','pitch3_moto_tmp','acc_x','acc_y',
    #                      'environment_tmp','int_tmp','pitch1_ng5_tmp','pitch2_ng5_tmp','pitch3_ng5_tmp',
    #                      'pitch1_ng5_DC','pitch2_ng5_DC','pitch3_ng5_DC']
    #
    #
    #
    # selected_features = ['wind_speed', 'generator_speed', 'power', 'yaw_position', 'int_tmp',
    #                      'pitch1_angle', 'pitch2_angle', 'pitch3_angle', 'pitch1_moto_tmp', 'pitch2_moto_tmp',
    #                      'pitch3_moto_tmp', 'environment_tmp','pitch1_ng5_tmp']


    # selected_features = ['AI_NAC_AveWindSpeed10min', 'AI_NAC_AveWindSpeed10s', 'AI_NAC_WindSpeed1',
    #                      'C_10Min_Aver_WindSpeed', 'AI_GEN_CoolAirTemp',
    #                      'C_15Min_Aver_WindSpeed', 'C_1Min_Aver_WindSpeed', 'AI_NAC_WindDir', 'AI_NAC_WindDir25s',
    #                      'AI_GBX_OilSumpTemp',
    #                      'C_TBN_10MinAveWindDir', 'AI_NAC_AirTemp', 'AI_NAC_CabTemp', 'AI_NAC_OutAirTemp',
    #                      'AI_TBN_RotorSpeed',
    #                      'AI_NAC_Position', 'AI_YAW_Speed', 'AI_PTC_Speed1', 'AI_PTC_Speed2', 'AI_PTC_Speed3',
    #                      'AI_PCS_MeasGenSpeed', 'AI_PTC_PosRef1', 'AI_PTC_PosRef2', 'AI_PTC_PosRef3', 'AI_NAC_VibX',
    #                      'AI_NAC_VibY', 'AI_PTC_DrvCurr1', 'AI_IPR_VoltNeutralL3L1']
    #
    #
    selected_features = ['AI_GEN_CoolAirTemp', 'AI_GBX_OilSumpTemp', 'AI_NAC_Position',

                         'C_TBN_10MinAveWindDir', 'AI_NAC_OutAirTemp', 'AI_NAC_AirTemp',
                         'C_10Min_Aver_WindSpeed','AI_NAC_CabTemp','AI_YAW_Speed']

    # 划分数据集
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)  # shuff

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    # 数据标准化
    zscore = preprocessing.StandardScaler()
    X_train = zscore.fit_transform(X_train)
    X_test = zscore.fit_transform(X_test)

    X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    print(X_train.shape)



    model = Sequential()

    # 偏航参数
    # model.add(LSTM(8, input_shape=(1, X_train.shape[2])))
    # model.add(Bidirectional(LSTM(16,input_shape=(1, X_train.shape[2]))))
    # 结冰参数

    model.add(LSTM(32, input_shape=(1, X_train.shape[2])))
    # model.add(SimpleRNN(128, input_shape=(1, X_train.shape[2])))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    metrics = Metrics()
    history = model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1, validation_data=(X_test, y_test),callbacks=[metrics])
    # history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=16, verbose=1)
    # model.fit(X_train, y_train, batch_size=16, epochs=2)
    model.summary()
    score = model.evaluate(X_test, y_test, batch_size=16)
    print(score)
    # 编译阶段引用自定义评价指标示例





# 绘制训练 & 验证的准确率值
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
#     #
#     # # 绘制训练 & 验证的损失值
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Model loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Test'], loc='upper left')
#     plt.show()