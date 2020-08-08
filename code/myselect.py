
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
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import AdaBoostClassifier  as ADB
from sklearn.feature_selection import SelectFromModel

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel






#方差选择法，返回值为特征选择后的数据
#参数threshold为方差的阈值




from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from xgboost import plot_importance
from matplotlib import pyplot


# 特征选择，选择相关度高的特征



if __name__=='__main__':
    # Step1 读取数据
    # data15 = pd.read_csv("../data/jb_train_set.csv")
    # data21 = pd.read_csv("../data/jb_test_set.csv")


    data15 = pd.read_csv("../data/ph_train_set.csv")
    data21 = pd.read_csv("../data/ph_test_set.csv")

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




    # selected_features1 = ['wind_speed', 'generator_speed', 'power', 'yaw_position', 'int_tmp',
    #                       'pitch1_angle', 'pitch2_angle', 'pitch3_angle', 'pitch1_moto_tmp', 'pitch2_moto_tmp',
    #                       'pitch3_moto_tmp', 'environment_tmp', 'pitch1_ng5_tmp']


    selected_features = ['AI_NAC_AveWindSpeed10min', 'AI_NAC_AveWindSpeed10s', 'AI_NAC_WindSpeed1',
                         'C_10Min_Aver_WindSpeed', 'AI_GEN_CoolAirTemp',
                         'C_15Min_Aver_WindSpeed', 'C_1Min_Aver_WindSpeed', 'AI_NAC_WindDir', 'AI_NAC_WindDir25s',
                         'AI_GBX_OilSumpTemp',
                         'C_TBN_10MinAveWindDir', 'AI_NAC_AirTemp', 'AI_NAC_CabTemp', 'AI_NAC_OutAirTemp',
                         'AI_TBN_RotorSpeed',
                         'AI_NAC_Position', 'AI_YAW_Speed', 'AI_PTC_Speed1', 'AI_PTC_Speed2', 'AI_PTC_Speed3',
                         'AI_PCS_MeasGenSpeed', 'AI_PTC_PosRef1', 'AI_PTC_PosRef2', 'AI_PTC_PosRef3', 'AI_NAC_VibX',
                         'AI_NAC_VibY', 'AI_PTC_DrvCurr1', 'AI_IPR_VoltNeutralL3L1','AI_IPR_RealPower']

    # selected_features = ['AI_GEN_CoolAirTemp', 'AI_GBX_OilSumpTemp', 'AI_NAC_Position',
    #                      'AI_NAC_VibY', 'AI_NAC_CabTemp',
    #                      'C_TBN_10MinAveWindDir', 'AI_NAC_OutAirTemp', 'AI_NAC_AirTemp', 'AI_NAC_VibX',
    #                      'C_15Min_Aver_WindSpeed',
    #                      'AI_YAW_Speed', 'C_10Min_Aver_WindSpeed', 'AI_NAC_AveWindSpeed10min', 'AI_PCS_MeasGenSpeed',
    #                      'AI_TBN_RotorSpeed']


    # selected_features = ['AI_GEN_CoolAirTemp', 'AI_GBX_OilSumpTemp', 'AI_NAC_Position',
    #
    #                      'C_TBN_10MinAveWindDir', 'AI_NAC_OutAirTemp', 'AI_NAC_AirTemp',
    #                      'C_10Min_Aver_WindSpeed','AI_NAC_CabTemp']



    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    print(X_train.shape)
    print(X_test.shape)


    # 数据标准化
    zscore = preprocessing.StandardScaler()
    X_train = zscore.fit_transform(X_train)
    X_test = zscore.fit_transform(X_test)

    # print(X_train[1:2, 0:10])
    # print(X_train[1:2, 10:20])
    # print(X_train[1:2, 20:27])
    #
    #
    # X_train = SelectFromModel(GBDT()).fit_transform(X_train, y_train)
    # print(X_train.shape)
    # print(X_train[1:2,:])


    model = XGBClassifier(n_estimators=300)
    # 模型训练
    model.fit(X_train , y_train)

    plot_importance(model)
    pyplot.show()

    y_pred = model.predict(X_test)
    # # 打印score
    print("XGB准确率",accuracy_score(y_test, y_pred))
    print("ROC,", roc_auc_score(y_test, y_pred))
    print("混淆矩阵", confusion_matrix(y_test, y_pred))
    print(classification_report(y_true=y_test, y_pred=y_pred))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, selected_features[indices[f]], importances[indices[f]]))




    clf = RF(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("RF准确率", accuracy_score(y_test, y_pred))
    print("ROC,",roc_auc_score(y_test, y_pred))
    print("混淆矩阵",confusion_matrix(y_test, y_pred))
    print(classification_report(y_true=y_test, y_pred=y_pred))

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, selected_features[indices[f]], importances[indices[f]]))



    # clf = ADB(n_estimators=100)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # print("ADB准确率", accuracy_score(y_test, y_pred))
    # print("ROC,", roc_auc_score(y_test, y_pred))
    # print("混淆矩阵", confusion_matrix(y_test, y_pred))
    # print(classification_report(y_true=y_test, y_pred=y_pred))
    #
    clf = GBDT(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("GBDT准确率", accuracy_score(y_test, y_pred))
    print("ROC,", roc_auc_score(y_test, y_pred))
    print("混淆矩阵", confusion_matrix(y_test, y_pred))
    print(classification_report(y_true=y_test, y_pred=y_pred))
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, selected_features[indices[f]], importances[indices[f]]))

    # clf = LR()
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # print("LG准确率", accuracy_score(y_test, y_pred))
    # print("ROC,", roc_auc_score(y_test, y_pred))
    # print("混淆矩阵", confusion_matrix(y_test, y_pred))
    # print(classification_report(y_true=y_test, y_pred=y_pred))

    # clf = SVM()
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # print("SVM准确率", accuracy_score(y_test, y_pred))
    # print("ROC,", roc_auc_score(y_test, y_pred))
    # print("混淆矩阵", confusion_matrix(y_test, y_pred))
    # print(classification_report(y_true=y_test, y_pred=y_pred))










