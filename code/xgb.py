import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split


# 模型测试

# Step1 读取数据
data15 = pd.read_csv("../data/15data.csv")
data21 = pd.read_csv("../data/21data.csv")
# 取特征和标签
X_train, y_train = data15.ix[:, 0:-1], data15.ix[:, -1]
X_test, y_test = data21.ix[:, 0:-1], data21.ix[:, -1]

# 划分数据集
# x, y = data15.ix[:, 0:-1], data15.ix[:, -1]
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.fit_transform(X_test)

# 特征选择
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
svc = SVC(kernel="linear")
dt = DecisionTreeClassifier()
rfecv = RFECV(estimator=dt, step=1, cv=StratifiedKFold(2), scoring='accuracy')
rfecv.fit(X_train, y_train)
print("Optimal number of features : %d" % rfecv.n_features_)
print("Ranking of features names: %s" % X_train.columns[rfecv.ranking_-1])
print("Ranking of features nums: %s" % rfecv.ranking_)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.savefig("feature.jpg")
plt.show()

# 特征对比图
import seaborn as sns
sns.pairplot(X_train, vars=["wind_speed","generator_speed", "power"],
             palette="husl"
            ,diag_kind="kde")
plt.savefig("duibi.jpg")


# 网格搜索随机森林最佳参数
# def GridsearchCV():
#     param_grid = [
#         {
#             'n_estimators': [i for i in range(500, 510)],
#             'oob_score': True,
#             'random_state': [i for i in range(30, 50)],
#             'min_samples_split': [i for i in range(2, 20)],
#             'max_depth': [i for i in range(100, 200)],
#         }]
#     rf_clf = RandomForestClassifier(max_depth=146, n_estimators=500,
#                                     max_leaf_nodes=2500, oob_score=True)
#     grid_search = GridSearchCV(rf_clf, param_grid, n_jobs=-1)
#     grid_search.fit(X, y)
#     grid_search.best_score_
#     grid_search.best_estimator_


# 使用随机森林分类器（直接使用网格搜索的最佳参数）
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(max_depth=146, n_estimators=2500,
                                max_leaf_nodes=2500, oob_score=True, random_state=30, n_jobs=-1)
rf_clf.fit(X_train, y_train)
y_predict = rf_clf.predict(X_test)
print(rf_clf.oob_score_)


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Normalization can be applied by setting `normalize = True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    print(cm)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("matrix.jpg")


from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

prediction = rf_clf.predict(X_test)
cm = confusion_matrix(y_test, prediction)
cm_plot_labels = ['normal', 'failure']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

# 评价
# precision & recall & f1-score
from sklearn.metrics import classification_report

print(classification_report(y_true=y_test, y_pred=prediction))
