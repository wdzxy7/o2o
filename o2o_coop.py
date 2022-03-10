import warnings
import numpy as np
import pandas as pd
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from datetime import date
from pandas import DataFrame
from chinese_calendar import is_workday
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')  # 不显示警告


def model_xgb(train, test):
    # xgb参数
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'gamma': 0.1,
              'min_child_weight': 1.1,
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'eta': 0.01,
               # 'tree_method': 'exact',
              'tree_method': 'gpu_hist',
              'seed': 0,
              'nthread': 12,
              'predictor': 'gpu_predictor',
              }
    print(params)
    x_data = train.drop(['label'], axis=1).copy()
    x_data.columns = test.columns.tolist()
    label_data = train['label'].copy()
    best_feature = KBest_select(x_data, label_data, 283)
    x_data = x_data[best_feature]
    test = test[best_feature]
    train_data = xgb.DMatrix(x_data, label=label_data)
    test_data = xgb.DMatrix(test)
    watchlist = [(train_data, 'train')]
    model = xgb.train(params, train_data, 5150, watchlist)  # 1500
    model.save_model('xgb_model.model')
    pred = model.predict(test_data)
    return pred


def model_xgb2(train, test):
    # xgb参数
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'tree_method': 'gpu_hist',
              'min_child_weight': 1.04,
              'max_depth': 2,  # 32
              'lambda': 11,  # 11
              'gamma': 0.22,
              'subsample': 0.7,
              'colsample_bytree': 0.763,
              'colsample_bylevel': 0.71,
              'eta': 0.10,
              'nthread': 5,
              'predictor': 'gpu_predictor',
              'verbosity': 1
              }
    print(params)
    x_data = train.drop(['label'], axis=1).copy()
    label_data = train['label'].copy()
    best_feature = KBest_select(x_data, label_data, 110)
    x_data = train[best_feature]
    test = test[best_feature]
    train_data = xgb.DMatrix(x_data, label=label_data)
    test_data = xgb.DMatrix(test)
    watchlist = [(train_data, 'train')]
    model = xgb.train(params, train_data, 1294, watchlist)
    pred = model.predict(test_data)
    return pred


def xgb_train():
    predict = model_xgb(Train_data, test_data)
    predict = pd.DataFrame(predict, columns=['predict'])
    result = pd.concat([test_front, predict], axis=1)
    # 保存
    result.to_csv('xgb_predict.csv', index=False, header=False)


def xgb_train2():
    predict = model_xgb2(Train_data, test_data)
    predict = pd.DataFrame(predict, columns=['predict'])
    result = pd.concat([test_front, predict], axis=1)
    # 保存
    result.to_csv('xgb_predict2.csv', index=False, header=False)


def KBest_select(x_data, y_data, feature_count):
    model = SelectKBest(chi2, k='all')
    model.fit_transform(x_data, y_data)
    scores = model.scores_
    indices = np.argsort(scores)[::-1]
    k_best_features = list(x_data.columns.values[indices[0:feature_count]])
    return k_best_features


def contact():
    res1 = pd.read_csv('xgb_predict.csv', header=None)
    res2 = pd.read_csv('xgb_predict2.csv', header=None)
    res1.columns = ['1', '2', '3', '4']
    res2.columns = ['1', '2', '3', '4']
    data = DataFrame([])
    data['1'] = res1['4'] * 0.3 + res2['4'] * 0.701
    print(data)
    front = res1[['1', '2', '3']]
    front = pd.concat([front, data], axis=1)
    front.to_csv('res.csv', index=False, header=False)


if __name__ == '__main__':
    t1 = pd.read_pickle('data/train1.pkl')
    t1 = DataFrame(t1)
    t2 = pd.read_pickle('data/train2.pkl')
    test_data = pd.read_pickle('data/test.pkl')
    Train_data = t1.append(t2)
    print('删除数据')
    # 去掉无用数据
    t_train = Train_data.copy()
    test_front = test_data[['User_id', 'Coupon_id', 'Date_received']].copy()
    labels = Train_data['label']
    Train_data.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], inplace=True, axis=1)
    test_data.drop(['User_id', 'Coupon_id', 'Date_received'], inplace=True, axis=1)
    Train_data['label'] = labels
    # 去重
    Train_data.drop_duplicates(keep='first', inplace=True)
    print('开始训练')
    test_data.fillna(0, inplace=True)
    Train_data = abs(Train_data)
    test_data = abs(test_data)
    test_front['Coupon_id'] = test_front['Coupon_id'].map(int)
    test_front['User_id'] = test_front['User_id'].map(int)
    test_front['Date_received'] = test_front['Date_received'].map(int)
    # -------------------------------------------------最终结果------------------------------------------------------
    xgb_train()
    # xgb_train2()
    # contact()