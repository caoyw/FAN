import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from imblearn.ensemble import EasyEnsemble
from imblearn.ensemble import BalanceCascade
from sklearn.linear_model import LogisticRegression as LG
from sklearn.ensemble import AdaBoostClassifier  as ADB


# 欠采样使数据平衡


# Step1 读取数据
# data15 = pd.read_csv("../data/L_15data.csv")
# data21 = pd.read_csv("../data/L_21data.csv")

data15 = pd.read_csv("../processed/data_3.csv")
data21 = pd.read_csv("../processed/data_4.csv")

selected_features = ['AI_NAC_AveWindSpeed10min', 'AI_NAC_AveWindSpeed10s', 'AI_NAC_WindSpeed1',  'C_10Min_Aver_WindSpeed','AI_GEN_CoolAirTemp',
                     'C_15Min_Aver_WindSpeed', 'C_1Min_Aver_WindSpeed', 'AI_NAC_WindDir', 'AI_NAC_WindDir25s','AI_GBX_OilSumpTemp',
                     'C_TBN_10MinAveWindDir', 'AI_NAC_AirTemp', 'AI_NAC_CabTemp','AI_NAC_OutAirTemp', 'AI_TBN_RotorSpeed',
                     'AI_NAC_Position', 'AI_YAW_Speed', 'AI_PTC_Speed1', 'AI_PTC_Speed2', 'AI_PTC_Speed3',
                     'AI_PCS_MeasGenSpeed', 'AI_PTC_PosRef1', 'AI_PTC_PosRef2','AI_PTC_PosRef3', 'AI_NAC_VibX',
                     'AI_NAC_VibY','AI_PTC_DrvCurr1','AI_IPR_VoltNeutralL3L1']

X_train = data21[selected_features]
print(X_train.shape)

selected_features = ['DI_YAW_ErrYawSpd']
y_train = data21[selected_features]

y_train = y_train.ix[:, -1]



# 取特征和标签
# X_train, y_train = data15.ix[:, 0:-1], data15.ix[:, -1]
# X_test, y_test = data21.ix[:, 0:-1], data21.ix[:, -1]

zscore = preprocessing.StandardScaler()
X_train = zscore.fit_transform(X_train)
# X_test = zscore.fit_transform(X_test)

# 划分数据集n_max_subset=4
bc = BalanceCascade(random_state=0,estimator=ADB(random_state=0),bootstrap=True)
# bc = BalanceCascade(random_state=0,estimator=ADB(random_state=0),n_max_subset=15)
x_resampled, y_resampled = bc.fit_sample(X_train , y_train)
# x_resampled, y_resampled = bc.fit_sample(X_test, y_test )

print(x_resampled.shape) # 打印输出集成方法处理后的x样本集概况
print(y_resampled.shape) # 打印输出集成方法处理后的y标签集概况


index_num = 1 # 设置抽样样本集索引
# x_resampled_t =pd.DataFrame(x_resampled[index_num],columns=['wind_speed','generator_speed','power','wind_direction','wind_direction_mean','yaw_position','yaw_speed','pitch1_angle','pitch2_angle','pitch3_angle','pitch1_speed','pitch2_speed','pitch3_speed','pitch1_moto_tmp','pitch2_moto_tmp','pitch3_moto_tmp','acc_x','acc_y','environment_tmp','int_tmp','pitch1_ng5_tmp','pitch2_ng5_tmp','pitch3_ng5_tmp','pitch1_ng5_DC','pitch2_ng5_DC','pitch3_ng5_DC'])
# # 将数据转换为数据框并命名列名
# y_resampled_t =pd.DataFrame(y_resampled[index_num],columns=['label']) # 将数据转换为数据框并命名列名
# # # 按列合并数据框
# EasyEnsemble_resampled = pd.concat([x_resampled_t,y_resampled_t], axis = 1)
# # 对label做分类汇总
# groupby_data_EasyEnsemble =EasyEnsemble_resampled.groupby('label').count()
# # 打印输出经过处理后的数据集样本分类分布
# print (groupby_data_EasyEnsemble)

train= np.column_stack((x_resampled[index_num],y_resampled[index_num]))
np.savetxt('ph_4_set.csv',train, delimiter = ',')
# np.savetxt('jb_test_set.csv',train, delimiter = ',')



