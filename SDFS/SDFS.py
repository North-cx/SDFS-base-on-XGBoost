
# generate regression datasets
class Bunch(dict):
    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_data():
    data_csv = pd.read_csv('data/data.csv'
                      , usecols = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                      , header = 0
                      , names = ["Datetime"
                                 , "total_value"
                                 , "VRF1_value"
                                 , "VRF2_value"
                                 , "lighting1_value"
                                 , "lighting2_value"
                                 , "temperature"
                                 , "enthalpy"
                                 , "relative_humidity"
                                 , "radiation"]
                      , index_col=[0]
                      , parse_dates=[0]
                      , encoding= 'ISO-8859-1')
    data = Bunch()
    data.data = _get_data(data_csv)
    data.target = _get_target(data_csv)
    data.DESCR = _get_descr(data_csv)
    data.feature_names = _get_feature_names()
    data.target_names = _get_target_names()

    return data

def _get_data(data):
    """
    获取特征值
    :return:
    """
    data_r = data.iloc[:, [5, 6, 7, 8]]
    data_np = np.array(data_r)
    return data_np


def _get_target(data):
    """
    获取目标值
    :return:
    """
    data_b = data.iloc[:, 4]
    data_np = np.array(data_b)
    return data_np


def _get_descr(data):
    """
    获取数据集描述
    :return:
    """
    text = "本数据集选取了2016-2019年夏季每日8:00:00-21:00:00的数据，样本数量：{}；" \
           "特征数量：{}；目标值数量：{}；无缺失数据" \
           "".format(data.index.size, 4, 1)
    return text


def _get_feature_names():
    """
    获取特征名字
    :return:
    """
    fnames = ["temperature"
              , "enthalpy"
              , "relative_humidity"
              , "radiation"]
    return fnames


def _get_target_names():
    """
    获取目标值名称
    :return:
    """
    tnames = ["lighting2_value"]
    return tnames

# prediction

from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import datetime


from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2, f_regression


data = load_data()
X = data.data
y = data.target

Xtrain,Xtest,Ytrain,Ytest = TTS(X,y,test_size=0.2,random_state=10)
print(X)
'''
fs = SelectKBest(score_func=f_regression, k='all') 
fs.fit(Xtrain, Ytrain) 
X_train_fs = fs.transform(Xtrain) 
X_test_fs = fs.transform(Xtest)

# what are scores for the features 
for i in range(len(fs.scores_)): 
    print('Feature %d: %f' % (i+1, fs.scores_[i])) 
# plot the scores 
plt.bar([i+1 for i in range(len(fs.scores_))], fs.scores_)
plt.xticks(np.arange(0, 5, 1))
plt.xlabel('Feature number')
plt.ylabel('p value')
plt.show()
'''

'''
reg = XGBR(n_estimators=100
           , max_depth = 5
           , learning_rate = 0.01
           , gamma = 0.1
           , min_child_weight = 3
           , subsample = 0.8
           , colsample_bytree = 0.8
           , reg_alpha = 0.05
           , reg_lambda = 0.05
           ).fit(Xtrain,Ytrain)
reg.predict(Xtest) #传统接口predict
reg.score(Xtest,Ytest) #你能想出这里应该返回什么模型评估指标么？
MSE(Ytest,reg.predict(Xtest))
reg.feature_importances_ #树模型的优势之一：能够查看模型的重要性分数，可以使用嵌入法进行特征选择

# reg = XGBR(n_estimators=100)
# CVS(reg,Xtrain,Ytrain,cv=5).mean()

def plot_learning_curve(estimator,title, X, y,
                        ax=None, #选择子图
                        ylim=None, #设置纵坐标的取值范围
                        cv=None, #交叉验证
                        n_jobs=None #设定索要使用的线程
                        ):

    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y
                                                            ,shuffle=True
                                                            ,cv=cv
                                                            # ,random_state=420
                                                            ,n_jobs=n_jobs)
    if ax == None:
        ax = plt.gca()
    else:
        ax = plt.figure()
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.grid() #绘制网格，不是必须
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-'
            , color="r",label="Training score")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-'
            , color="g",label="Test score")
    ax.legend(loc="best")
    return ax

cv = KFold(n_splits=5, shuffle = True, random_state=42)
plot_learning_curve(XGBR(n_estimators=100,random_state=420)
                    ,"XGB",Xtrain,Ytrain,ax=None,cv=cv)
plt.show()
'''