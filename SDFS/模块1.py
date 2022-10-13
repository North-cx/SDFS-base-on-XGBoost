
### Preparation

# Load Modules
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Set Plotting Styles
plt.style.use('ggplot') 
matplotlib.rcParams['figure.figsize'] = (18, 18) #显示图像的最大范围

# Load Data
Unit_df = pd.read_csv('data/data.csv'
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

# Create Features
def get_time_state(time):
    if time in range(8, 18):
        return 1
    else:
        return 0
    
def get_work_state(DayOfWeek):
    if DayOfWeek in range(1, 6):
        return 1
    else:
        return 0

def create_features(df):
    df['Date'] = df.index
    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['DayOfMonth'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['DayOfYearFloat'] = df['DayOfYear'] + df['Hour'] / 24
    df['Time_state'] = df.apply(lambda x: get_time_state(x['Hour']), axis=1)
    df['Working_state'] = df.apply(lambda x: get_work_state(x['DayOfWeek']), axis=1)
    df.drop('Date', axis=1, inplace=True)
    return df
Unit_df = create_features(Unit_df)

### Feature Engineering

# maximum Pearson correlation coefficient
def Pearson_correlation():
    for i in ["VRF1_value", "VRF2_value", "lighting1_value", "lighting2_value"]:
        corr = Unit_df.loc[:,["lighting2_value",i]].corr(method='pearson')
        print(i)
        print(corr)

#Pearson_correlation()
    
def create_training_sample(datatype):
    # Construction of historical training data set in the normalbranch, this step takes the historical energy consumption data as a
    # sample in a day and the 15-dimensional training sample TP 
    data = pd.DataFrame(columns=[list(map(str, list(range(7, 22))))])
    lighting1_columns_index = Unit_df.columns.get_loc(datatype)
    for i in range(Unit_df.shape[0]//15):
        i*=15
        lis = []
        #lis.append(Unit_df.index[i].strftime("%Y-%m-%d"))
        for j in range(15):
            lis.append(Unit_df.iloc[i+j,lighting1_columns_index])
        data.loc[i//15] = lis
    data.set_index(Unit_df.index[0::15], inplace=True)
    return data
        
lighting1_value = create_training_sample('lighting1_value')
lighting2_value = create_training_sample('lighting2_value')

def Distance_calculation(dataname, a, b):
    # calculation certain data's Euclidean Distance between date a and date b 
    sum = 0
    for i in (dataname.iloc[a][0:] - dataname.iloc[b][0:]):
        sum += pow(i, 2)
    return pow(sum, 0.5)

def bubbleSort(arr):
    n = len(arr)
 
    # 遍历所有数组元素
    for i in range(n):
 
        # Last i elements are already in place
        for j in range(0, n-i-1):
 
            if arr[j][1] < arr[j+1][1] :
                arr[j], arr[j+1] = arr[j+1], arr[j]
            
def Find_historical_similar_dataset(dataname, a, K):
    #K is not finished
    lis = list(range(dataname.shape[0]))
    lis.remove(a)
    dateset1 = []
    for n in range(K):
        dateset1.append([0, 1e10])
    for i in lis:
        value = Distance_calculation(dataname, a, i) 
        for j in dateset1:
            if value < j[1]:
                j[0] = dataname.iloc[i].name
                j[1] = value
                bubbleSort(dateset1)
                break
    dateset2 = []
    for i in dateset1:
        dateset2.append(i[0])
    return dateset2
 
def get_SECF(datanameA, datanameB, K):
    #The energy consumption at the same time as setA can be found from the historical data of the abnormal branch, and is denoted as setB.
    #if K != 3 a,b,c.... should be insteaded by other variable whose numbers equals K
    SECF = []
    for i in range(datanameA.shape[0]):
        dateset = Find_historical_similar_dataset(datanameA, i, K)
        F = []
        for j in dateset:
            F.append(list(datanameB.loc[j]))
        SECF += [(a + b + c) / 3 for a, b, c in zip(F[0], F[1], F[2])]
    return SECF

lighting2_value_SECF = get_SECF(lighting1_value, lighting2_value, 3)
Unit_df.insert(Unit_df.shape[1], 'lighting2_value_SECF', lighting2_value_SECF)
Unit_df.to_csv('data/df.csv',sep=',',index=True,header=True)

#correlation coefficient heatmap
for i in ["VRF1_value", "VRF2_value", "lighting1_value", "lighting2_value", 'lighting2_value_SECF']:
        corr = Unit_df.loc[:,["lighting2_value",i]].corr(method='pearson')
        print(i)
        print(corr)
        
def showcov(df):
    dfData = df.corr()
    plt.subplots(figsize=(6, 6)) # 设置画面大小
    sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues")
    #matplotlib.rcParams.update({'font.size': 10})
    plt.tight_layout() # Make the text inside the picture
    plt.savefig('images/BluesStateRelation.png')
    plt.show()

#showcov(Unit_df.loc[:,["total_value", "VRF1_value", "VRF2_value", "lighting1_value", "lighting2_value", 'lighting2_value_SECF']])

### Maching learning















#Feature_importance
from xgboost import plot_importance
def Feature_importance():
    xgbr_y_predict=reg.predict(Xtest)

    plot_importance(reg, height = 0.5)

    # fontsize
    #plt.xticks(fontsize=30)
    #plt.yticks(fontsize=30)
    #matplotlib.rcParams.update({'font.size': 30})

    plt.tight_layout() # Make the text inside the picture

    plt.savefig('images/Feature_importance_plot.png')

    plt.show()