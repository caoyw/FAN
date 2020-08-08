
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV



from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC as SVM
from sklearn.linear_model import LogisticRegression as LG
from sklearn.ensemble import AdaBoostClassifier  as ADB
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from xgboost import plot_importance
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
import time
# 多个机器学习算法整合

if __name__=='__main__':
    # Step1 读取数据
    data15 = pd.read_csv("../data/jb_train_set.csv")
    data21 = pd.read_csv("../data/jb_test_set.csv")


    # data15 = pd.read_csv("../data/ph_train_set.csv")
    # # data21 = pd.read_csv("../data/ph_test_set.csv")

    data15 = pd.read_csv("../data/ph_4_set.csv")
    data21 = pd.read_csv("../data/ph_3_set.csv")

    # data15 = pd.read_csv("../data/L_15data.csv")
    # data21 = pd.read_csv("../data/L_21data.csv")


    # 取特征和标签
    X_train, y_train = data15.ix[:, 0:-1], data15.ix[:, -1]
    X_test, y_test = data21.ix[:, 0:-1], data21.ix[:, -1]

    # selected_features = ['wind_speed','generator_speed','power','wind_direction',
    #                      'wind_direction_mean','yaw_position','yaw_speed','pitch1_angle',
    #                      'pitch2_angle','pitch3_angle','pitch1_speed','pitch2_speed','pitch3_speed',
    #                      'pitch1_moto_tmp','pitch2_moto_tmp','pitch3_moto_tmp','acc_x','acc_y',
    #                      'environment_tmp','int_tmp','pitch1_ng5_tmp','pitch2_ng5_tmp','pitch3_ng5_tmp',
    #                      'pitch1_ng5_DC','pitch2_ng5_DC','pitch3_ng5_DC']
    # #
    # selected_features = ['wind_speed', 'generator_speed', 'power','yaw_position','int_tmp',
    #                      'pitch1_angle','pitch2_angle', 'pitch3_angle', 'pitch1_moto_tmp', 'pitch2_moto_tmp',
    #                      'pitch3_moto_tmp','environment_tmp',  'pitch1_ng5_tmp']



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
    selected_features = ['AI_GEN_CoolAirTemp', 'AI_GBX_OilSumpTemp', 'AI_NAC_Position',

                         'C_TBN_10MinAveWindDir', 'AI_NAC_OutAirTemp', 'AI_NAC_AirTemp',
                         'C_10Min_Aver_WindSpeed','AI_NAC_CabTemp','AI_YAW_Speed']

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)# shuff

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    print(X_train.shape)
    print(X_test.shape)


    # 数据标准化
    zscore = preprocessing.StandardScaler()
    X_train = zscore.fit_transform(X_train)
    X_test = zscore.fit_transform(X_test)

    starttime = time.time()
    model = XGBClassifier( )
    param_dist = {
                  'max_depth': range(3, 10, 1),
                  'learning_rate': np.linspace(0.01, 2, 20)
                  }

    # model = GridSearchCV( model, param_dist, cv=3, scoring='neg_log_loss', n_jobs=-1)
    # 模型训练
    model.fit(X_train , y_train)
    y_pred = model.predict(X_test)
    print("XGB准确率:",accuracy_score(y_test, y_pred))
    print("精准率:", precision_score(y_test, y_pred))
    print("召回率:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("ROC:", roc_auc_score(y_test, y_pred))
    confusion_mat = confusion_matrix(y_test, y_pred)
    endtime = time.time()
    totaltime = endtime - starttime
    print("XGB的时间:", totaltime)
    print("   ")

    # ax = model.plot_tree(model, tree_index=1, figsize=(20, 8), show_info=['split_gain'])
    # plt.show()


    starttime = time.time()
    clf = GBDT(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("GBDT准确率:", accuracy_score(y_test, y_pred))
    print("精准率:", precision_score(y_test, y_pred))
    print("召回率:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("ROC:", roc_auc_score(y_test, y_pred))
    endtime = time.time()
    totaltime = endtime - starttime
    print("GBDT的时间:", totaltime)
    print("   ")

    # 画ROC
    # false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    # roc_auc = auc(false_positive_rate, true_positive_rate)
    # plt.title('Receiver Operating Characteristic')
    # plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
    # # plt.plot(false_positive_rate1, true_positive_rate1, 'b', label='AUC2 = %0.2f' % roc_auc1)
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([-0.1, 1.0])
    # plt.ylim([-0.1, 1.05])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()



    # clf = RF(n_estimators=100)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # print("ET准确率", accuracy_score(y_test, y_pred))
    # print("精准率:", precision_score(y_test, y_pred))
    # print("召回率:", recall_score(y_test, y_pred))
    # print("F1:", f1_score(y_test, y_pred))
    # print("ROC:", roc_auc_score(y_test, y_pred))
    # print("   ")
    starttime = time.time()
    clf = ADB(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Adboost准确率:",accuracy_score(y_test, y_pred))
    print("精准率:", precision_score(y_test, y_pred))
    print("召回率:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("ROC:", roc_auc_score(y_test, y_pred))
    endtime = time.time()
    totaltime = endtime - starttime
    print("adb的时间:", totaltime)
    print("   ")

    starttime = time.time()
    clf = LG()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("LG准确率:",accuracy_score(y_test, y_pred))
    print("精准率:", precision_score(y_test, y_pred))
    print("召回率:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("ROC:", roc_auc_score(y_test, y_pred))
    endtime = time.time()
    totaltime = endtime - starttime
    print("lg的时间:", totaltime)
    print("   ")


    clf = SVM()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("SVM准确率:",accuracy_score(y_test, y_pred))
    print("精准率:", precision_score(y_test, y_pred))
    print("召回率:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("ROC:", roc_auc_score(y_test, y_pred))
    endtime = time.time()
    totaltime = endtime - starttime
    print("SVM的时间:", totaltime)











    # learning_rate = [ 0.1, 0.3,1]
    # param_grid = dict(learning_rate=learning_rate)
    # kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    # grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    # grid_result = grid_search.fit(X_train , y_train)
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #
    # best_estimator = grid_result.best_estimator_
    # best_score = grid_result.best_score_
    # print(best_estimator)
    # print(best_score)


    # 网络搜索法
    # param_dist = {
    #     # 'n_estimators': range(80, 200, 4),
    #     'learning_rate' : [0.1, 0.3, 1]
    #     # 'max_depth': range(2, 15, 1),
    #     # 'learning_rate': np.linspace(0.01, 2, 20),
    #     # 'subsample': np.linspace(0.7, 0.9, 20),
    #     # 'colsample_bytree': np.linspace(0.5, 0.98, 10),
    #     # 'min_child_weight': range(1, 9, 1)
    # }
    # kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    # grid = GridSearchCV(model, param_dist, cv=kfold, scoring='neg_log_loss', n_jobs=-1)
    #
    # # 在训练集上训练
    # grid.fit(X_train , y_train)
    # # 返回最优的训练器
    # best_estimator = grid.best_estimator_
    # best_score = grid.best_score_
    # print(best_estimator)
    # print(best_score)





    # 画ROC
    # false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    # roc_auc = auc(false_positive_rate, true_positive_rate)
    # plt.title('Receiver Operating Characteristic')
    # plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f' % roc_auc)
    # # plt.plot(false_positive_rate1, true_positive_rate1, 'b', label='AUC2 = %0.2f' % roc_auc1)
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([-0.1, 1.0])
    # plt.ylim([-0.1, 1.05])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()