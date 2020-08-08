import os


import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split
# from imblearn.ensemble import EasyEnsemble
# from imblearn.ensemble import BalanceCascade
from sklearn.linear_model import LogisticRegression
# Step1 读取数据
nordata = pd.read_csv("../data/ph_test_set.csv")
print(nordata.shape)

# 各种测试


x, y = nordata.ix[:, 0:-1], nordata.ix[:, -1]
# x, y = faidata.ix[:, 1:27], faidata.ix[:, 28]


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=42)# shuff
# train= np.column_stack((X_train,y_train))
# np.savetxt('phtrain_set.csv',train, delimiter = ',')
test = np.column_stack((X_test, y_test))
print(X_test.shape)
np.savetxt('phtest_set.csv', test, delimiter = ',')


# Step2 数据预处理,这里为了节约时间，仅使用百分之10的数据作训练和预测（其实更多比例也不会特别费时）
from sklearn.model_selection import train_test_split
# 随机选择百分之10的数据
# X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.9, random_state=666, shuffle = True)# shuffle默认为True
# # 在选择的数据中，选择2/3作为训练集，1/3作为测试集
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=666, shuffle = False)# shuffle默认为True



# model_EasyEnsemble = EasyEnsemble() # 建立EasyEnsemble模型对象
# x_EasyEnsemble_resampled, y_EasyEnsemble_resampled = model_EasyEnsemble.fit_sample(x, y) # 输入数据并应用集成方法处理
# print(x_EasyEnsemble_resampled.shape) # 打印输出集成方法处理后的x样本集概况
# print(y_EasyEnsemble_resampled.shape) # 打印输出集成方法处理后的y标签集概况
#
#
# index_num = 9 # 设置抽样样本集索引
# x_EasyEnsemble_resampled_t =pd.DataFrame(x_EasyEnsemble_resampled[index_num],columns=['wind_speed','generator_speed','power','wind_direction','wind_direction_mean','yaw_position','yaw_speed','pitch1_angle','pitch2_angle','pitch3_angle','pitch1_speed','pitch2_speed','pitch3_speed','pitch1_moto_tmp','pitch2_moto_tmp','pitch3_moto_tmp','acc_x','acc_y','environment_tmp','int_tmp','pitch1_ng5_tmp','pitch2_ng5_tmp','pitch3_ng5_tmp','pitch1_ng5_DC','pitch2_ng5_DC','pitch3_ng5_DC'])
# # 将数据转换为数据框并命名列名
# y_EasyEnsemble_resampled_t =pd.DataFrame(y_EasyEnsemble_resampled[index_num],columns=['label']) # 将数据转换为数据框并命名列名
# EasyEnsemble_resampled = pd.concat([x_EasyEnsemble_resampled_t,y_EasyEnsemble_resampled_t], axis = 1) # 按列合并数据框
# groupby_data_EasyEnsemble =EasyEnsemble_resampled.groupby('label').count() # 对label做分类汇总
# # print (groupby_data_EasyEnsemble) # 打印输出经过EasyEnsemble处理后的数据集样本分类分布