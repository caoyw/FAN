# mnist attention
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
np.random.seed(1337)
import pandas as pd
# from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import numpy
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score



class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []


    def on_epoch_end(self, epoch, logs={}):
        val_predict = (numpy.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        print("  f1:",_val_f1)
        print("recal:", _val_recall)
        print("pre:",_val_precision)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        return





if __name__=='__main__':


    # Step1 读取数据
    data15 = pd.read_csv("../data/jb_train_set.csv")
    data21 = pd.read_csv("../data/jb_test_set.csv")


    data15 = pd.read_csv("../data/ph_train_set.csv")
    data21 = pd.read_csv("../data/ph_test_set.csv")


    data15 = pd.read_csv("../data/ph_4_set.csv")
    data21 = pd.read_csv("../data/ph_3_set.csv")

    # 取特征和标签
    X_train, y_train = data15.ix[:, 0:-1], data15.ix[:, -1]
    X_test, y_test = data21.ix[:, 0:-1], data21.ix[:, -1]


    # selected_features = ['wind_speed', 'generator_speed', 'power', 'wind_direction',
    #                      'wind_direction_mean', 'yaw_position', 'yaw_speed', 'pitch1_angle',
    #                      'pitch2_angle', 'pitch3_angle', 'pitch1_speed', 'pitch2_speed', 'pitch3_speed',
    #                      'pitch1_moto_tmp', 'pitch2_moto_tmp', 'pitch3_moto_tmp', 'acc_x', 'acc_y',
    #                      'environment_tmp', 'int_tmp', 'pitch1_ng5_tmp', 'pitch2_ng5_tmp', 'pitch3_ng5_tmp',
    #                      'pitch1_ng5_DC', 'pitch2_ng5_DC', 'pitch3_ng5_DC']
    #
    #
    # selected_features = ['wind_speed', 'generator_speed', 'power', 'yaw_position', 'int_tmp',
    #                      'pitch1_angle', 'pitch2_angle', 'pitch3_angle', 'pitch1_moto_tmp', 'pitch2_moto_tmp',
    #                      'pitch3_moto_tmp', 'environment_tmp', 'pitch1_ng5_tmp']



    # selected_features = ['AI_NAC_AveWindSpeed10min', 'AI_NAC_AveWindSpeed10s', 'AI_NAC_WindSpeed1',
    #                      'C_10Min_Aver_WindSpeed', 'AI_GEN_CoolAirTemp',
    #                      'C_15Min_Aver_WindSpeed', 'C_1Min_Aver_WindSpeed', 'AI_NAC_WindDir', 'AI_NAC_WindDir25s',
    #                      'AI_GBX_OilSumpTemp',
    #                      'C_TBN_10MinAveWindDir', 'AI_NAC_AirTemp', 'AI_NAC_CabTemp', 'AI_NAC_OutAirTemp',
    #                      'AI_TBN_RotorSpeed',
    #                      'AI_NAC_Position', 'AI_YAW_Speed', 'AI_PTC_Speed1', 'AI_PTC_Speed2', 'AI_PTC_Speed3',
    #                      'AI_PCS_MeasGenSpeed', 'AI_PTC_PosRef1', 'AI_PTC_PosRef2', 'AI_PTC_PosRef3', 'AI_NAC_VibX',
    #                      'AI_NAC_VibY', 'AI_PTC_DrvCurr1', 'AI_IPR_VoltNeutralL3L1']






    selected_features = ['AI_GEN_CoolAirTemp', 'AI_GBX_OilSumpTemp', 'AI_NAC_Position',

                         'C_TBN_10MinAveWindDir', 'AI_NAC_OutAirTemp', 'AI_NAC_AirTemp',
                         'C_10Min_Aver_WindSpeed','AI_NAC_CabTemp','AI_YAW_Speed']

    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)  # shuff


    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    zscore = preprocessing.StandardScaler()
    X_train = zscore.fit_transform(X_train)
    X_test = zscore.fit_transform(X_test)

    X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    print(X_train.shape)

    # first way attention
    def attention_3d_block(inputs):

        # input_dim = int(inputs.shape[2])
        a = Permute((2, 1))(inputs)
        a = Dense(1, activation='sigmoid')(a)
        a_probs = Permute((2, 1), name='attention_vec')(a)
        output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
        return output_attention_mul


    # build RNN model with attention
    inputs = Input(shape=(1, X_train.shape[2]))
    # drop1 = Dropout(0.5)(inputs)
    # 偏航参数
    out = Conv1D(16, 1, activation='relu')(inputs)
    out = Conv1D(16, 1, activation='relu')(out)
    # 结冰参数
    # out = Conv1D(64, 1, activation='relu')(inputs)
    # out = Conv1D(64, 1, activation='relu')(out)
    out = Dropout(0.5)(out)
    drop1 = MaxPooling1D(1)(out)
    drop1 = Dropout(0.5)(out)

    # 偏航参数
    lstm_out = Bidirectional(LSTM(32, return_sequences=True), name='bilstm')(drop1)
    # lstm_out = LSTM(32, return_sequences=True)(drop1)
    # 结冰参数
    # lstm_out = Bidirectional(LSTM(16, return_sequences=True), name='bilstm')(drop1)
    lstm_out = Dropout(0.5)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)
    attention_flatten = Flatten()(attention_mul)
    drop2 = Dropout(0.5)(attention_flatten)
    output = Dense(1, activation='sigmoid')(drop2)

    model = Model(inputs=inputs, outputs=output)
    # second way attention
    # inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
    # units = 32
    # activations = LSTM(units, return_sequences=True, name='lstm_layer')(inputs)
    # attention = Dense(1, activation='tanh')(activations)
    # attention = Flatten()(attention)
    # attention = Activation('softmax')(attention)
    # attention = RepeatVector(units)(attention)
    # attention = Permute([2, 1], name='attention_vec')(attention)
    # attention_mul = merge([activations, attention], mode='mul', name='attention_mul')
    # out_attention_mul = Flatten()(attention_mul)
    # output = Dense(10, activation='sigmoid')(out_attention_mul)
    # model = Model(inputs=inputs, outputs=output)


    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    print('Training------------')
    # model.fit(X_train, y_train, epochs=10, batch_size=16)
    metrics = Metrics()
    history = model.fit(X_train, y_train, epochs=2, batch_size=16, validation_data=(X_test, y_test), callbacks=[metrics])
    print('Testing--------------')
    loss, accuracy = model.evaluate(X_test, y_test)
    print('test loss:', loss)
    print('test accuracy:', accuracy)

# 绘制训练 & 验证的准确率值
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title('Model accuracy')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Test'], loc='upper left')
#     plt.show()
#     #
#     # # 绘制训练 & 验证的损失值
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Model loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Test'], loc='upper left')
#     plt.show()