from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import AdaBoostClassifier as ADA


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from itertools import chain
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import precision_recall_fscore_support


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from numpy import log1p
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA

from sklearn import preprocessing
from xgboost import XGBClassifier as XGB
# from catboost import CatBoostClassifier
# import lightgbm as lgb
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.svm import SVC as SVM
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

from keras.layers import *
from keras.models import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from xgboost import plot_importance
from matplotlib import pyplot

import numpy
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    def attention_3d_block(inputs):

        # input_dim = int(inputs.shape[2])
        a = Permute((2, 1))(inputs)
        a = Dense(1, activation='sigmoid')(a)
        a_probs = Permute((2, 1), name='attention_vec')(a)
        output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
        return output_attention_mul



    data15 = pd.read_csv("../data/jb_train_set.csv")
    data21 = pd.read_csv("../data/jb_test_set.csv")



    data15 = pd.read_csv("../data/ph_4_set.csv")
    data21 = pd.read_csv("../data/ph_3_set.csv")

    # 取特征和标签
    x_train, y_train = data15.ix[:, 0:-1], data15.ix[:, -1]
    x_test, y_test = data21.ix[:, 0:-1], data21.ix[:, -1]

    # selected_features = ['wind_speed', 'generator_speed', 'power', 'wind_direction',
    #                      'wind_direction_mean', 'yaw_position', 'yaw_speed', 'pitch1_angle',
    #                      'pitch2_angle', 'pitch3_angle', 'pitch1_speed', 'pitch2_speed', 'pitch3_speed',
    #                      'pitch1_moto_tmp', 'pitch2_moto_tmp', 'pitch3_moto_tmp', 'acc_x', 'acc_y',
    #                      'environment_tmp', 'int_tmp', 'pitch1_ng5_tmp', 'pitch2_ng5_tmp', 'pitch3_ng5_tmp',
    #                      'pitch1_ng5_DC', 'pitch2_ng5_DC', 'pitch3_ng5_DC']
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
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=42)  # shuff

    x_train = x_train[selected_features]
    x_test = x_test[selected_features]

    # 数据标准化
    zscore = preprocessing.StandardScaler()
    x_train = zscore.fit_transform(x_train)
    x_test = zscore.fit_transform(x_test)


    print(x_train.shape)
    # 结冰
    # model = Sequential()
    # model.add(Conv1D(16, 1, activation='relu', input_shape=(1, x_train.shape[2])))
    # model.add(Conv1D(16, 1, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(MaxPooling1D(1))
    # model.add(LSTM(32))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))
    # model1 = model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # 偏航
    model = Sequential()
    inputs = Input(shape=(1, x_train.shape[1]))
    out = Conv1D(16, 1, activation='relu')(inputs)
    out = Conv1D(16, 1, activation='relu')(out)
    out = Dropout(0.5)(out)
    drop1 = MaxPooling1D(1)(out)
    drop1 = Dropout(0.5)(out)
    lstm_out = LSTM(32, return_sequences=True)(drop1)
    lstm_out = Dropout(0.5)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)
    attention_flatten = Flatten()(attention_mul)
    drop2 = Dropout(0.5)(attention_flatten)
    output = Dense(1, activation='sigmoid')(drop2)
    model1 = Model(inputs=inputs, outputs=output)
    model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])









    model2 = Sequential()
    model2.add(LSTM(units=32, input_shape=(1, x_train.shape[1])))
    model2.add(Dense(1, activation='sigmoid'))
    model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    ### 第一层模型
    clfs = [

         # GBDT(n_estimators=100),

        # RF(n_estimators=100),

        model1,
        XGB(n_estimators=100)

        # SVM()

    ]

    X_train_stack  = np.zeros((x_train.shape[0], len(clfs)))
    X_test_stack = np.zeros((x_test.shape[0], len(clfs)))


    # 5折stacking
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)
    print("longding...")
    for i,clf in enumerate(clfs):
        # print("分类器：{}".format(clf))
        X_stack_test_n = np.zeros((x_test.shape[0], n_folds))
        x_test_lstm = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
        for j,(train_index,test_index) in enumerate(skf.split(x_train,y_train)):

                    if i == 1:
                        tr_x = x_train[train_index]
                        # print(tr_x)
                        # print(tr_x.shape)
                        tr_y = y_train[train_index]
                        clf.fit(tr_x, tr_y)
                        # 生成stacking训练数据集
                        # X_train_stack [test_index, i] = clf.predict_proba(X_train[test_index])[:,1]
                        # X_stack_test_n[:,j] = clf.predict_proba(X_test)[:,1]
                        X_train_stack[test_index, i] = clf.predict(x_train[test_index])
                        X_stack_test_n[:, j] = clf.predict(x_test)
                    else:
                        print(train_index.shape)
                        print(train_index)
                        tr_x = x_train[train_index]
                        tr_y = y_train[train_index]
                        tr_x = tr_x.reshape(tr_x.shape[0], 1, tr_x.shape[1])
                        test_x = x_train[test_index]
                        test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])
                        clf.fit(tr_x, tr_y, epochs=5, batch_size=16)
                        temp = clf.predict(test_x)
                        X_train_stack[test_index, i] = temp.shape[0]
                        temp = clf.predict(x_test_lstm)
                        X_stack_test_n[:, j] = temp.shape[0]

        #生成stacking测试数据集
        X_test_stack[:,i] = X_stack_test_n.mean(axis=1)





    ###第二层模型LR
    clf_second = XGB()

    # clf_second = DecisionTreeClassifier()
    clf_second.fit(X_train_stack,y_train)
    pred = clf_second.predict_proba(X_test_stack)[:,1]
    # print('准确率1：', accuracy_score(y_test, pred))
    # auc = roc_auc_score(y_test,pred)#0.9946
    # print( 'auc：', auc)

    #准确率
    y_pred = clf_second.predict(X_test_stack)
    print( '准确率：', accuracy_score(y_test,y_pred))

    print("精准率:", precision_score(y_test, y_pred))
    print("召回率:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("ROC:", roc_auc_score(y_test, y_pred))

    # precision = precision_score(y_test, pred)
    # print('Precision:\t', precision)
    # recall = recall_score(y_test, pred)
    # print('Recall:  \t', recall)
    # print('f1 score: \t', f1_score(y_test, pred))






    # # ###GBDT分类器
    # clf_1 = clfs[0]
    # clf_1.fit(x_train,y_train)
    # # pred_1 = clf_1.predict_proba(X_test)[:,1]
    # # roc_auc_score(y_test,pred_1)#0.9922
    # pred_1 = clf_1.predict(x_test)
    # print( 'GBDTAccuracy：', accuracy_score(y_test,pred_1))
    # #
    # # ###随机森林分类器
    # clf_2 = clfs[1]
    # clf_2.fit(x_train,y_train)
    # # pred_2 = clf_2.predict_proba(X_test)[:,1]
    # # roc_auc_score(y_test,pred_2)#0.9944
    # pred_2 = clf_2.predict(x_test)
    # print( 'RFAccuracy：', accuracy_score(y_test,pred_2))
    # #
    # #
    # # ###ExtraTrees分类器
    # clf_3 = clfs[2]
    # clf_3.fit(x_train,y_train)
    # # pred_3 = clf_3.predict_proba(X_test)[:,1]
    # # roc_auc_score(y_test,pred_3)#0.9930
    # pred_3 = clf_3.predict(x_test)
    # print( 'ETAccuracy：', accuracy_score(y_test,pred_3))
    #
    #
    #
    #
    # gbm = lgb.LGBMRegressor(learning_rate=0.03, n_estimators=200, max_depth=8)
    # gbm.fit(x_train, y_train.ravel())
    # # 预测结果
    # y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration_)
    # y_pre = [int(item > 0.5) for item in y_pred]
    # print('gbm：', accuracy_score(y_test, y_pre))
    #
    # xgb = XGBClassifier()
    # xgb.fit(x_train, y_train.ravel())
    # # 预测结果
    # y_pred = xgb.predict(x_test)
    # print('XGB：', accuracy_score(y_test, y_pred))
    #
    # cat_features = [0, 1]
    # cat = CatBoostClassifier(iterations=2, depth=2, learning_rate=0.5, loss_function='Logloss', logging_level='Verbose')
    # cat.fit(x_train, y_train.ravel())
    # # 预测结果
    # y_pred = cat.predict(x_test)
    # print('CAT：', accuracy_score(y_test, y_pred))














    # SVM
    # clf = SVC(C=0.2, kernel='rbf', gamma=2, decision_function_shape='ovr')
    # clf.fit(x_train, y_train)
    # pred_5 = clf.predict(x_test)
    # print( 'SVMAccuracy：', accuracy_score(y_test,pred_5))

