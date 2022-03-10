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


def judge(x):
    # 0没用1用
    if x == 0:
        return 0
    else:
        return 1


def change_rate(x):
    rates = str(x).split(':')
    if len(rates) == 1:
        discount = float(x)
    else:
        discount = 1 - float(rates[1]) / float(rates[0])
    return round(discount, 5)


def change_dis(x, avg_dis):
    if x == '-999':
        return avg_dis
    else:
        return x


def count_distinct(x):
    x = set(x)
    return len(x)


def get_cou_kind(x):
    rates = str(x).split(':')
    if len(rates) == 1:
        return 0
    else:
        return 1


def get_min_use(x):
    rates = str(x).split(':')
    if len(rates) == 1:
        return 0
    else:
        return int(rates[0])


def front_get(x):
    x = x.sort_values(by='temp')
    res = []
    if len(x) > 2:
        for i in range(len(x)):
            if i == 0:
                x.iloc[i]['temp'] = 0
                res.append(0)
            else:
                k = x.iloc[i]['temp'] - x.iloc[i - 1]['temp']
                res.append(k.days)
        x['temp'] = res
    else:
        x['temp'] = 0
    return x


def get_use_day(x):
    if x['use'] == 1:
        k = (x['Date'] - x['Date_received']).days
        return k
    else:
        return 0


def get_avg_days(x):
    x = DataFrame(x)
    x.columns = ['Date']
    x = x.sort_values(by='Date')
    s = 0
    if len(x) > 2:
        for i in range(len(x)):
            if i == 0:
                continue
            else:
                k = x.iloc[i]['Date'] - x.iloc[i - 1]['Date']
                s += k.days
    else:
        return 0
    return s / len(x)


def change(x):
    if x == np.inf:
        return 0
    else:
        return x


def get_type(x):
    if ':' in str(x):
        return 1
    else:
        return 0


def get_max_threshold(x):
    rates = str(x).split(':')
    if len(rates) == 1:
        threshold = 0
    else:
        threshold = float(rates[0])
    return threshold


def get_discount(x):
    rates = str(x).split(':')
    if len(rates) == 1:
        threshold = 0
    else:
        threshold = float(rates[1])
    return threshold


def get_feature(label_data, extract_data):
    data_index = label_data.index
    # 未发放的设为0
    extract_data['Coupon_id'] = extract_data['Coupon_id'].fillna(0)
    # 用平均距离替代未知距离
    extract_data['Distance'] = extract_data['Distance'].fillna('-999')
    hav_dis = extract_data[(extract_data.Distance != '-999')]
    avg_dis = np.mean(hav_dis['Distance'])
    avg_dis = np.round(avg_dis, 1)
    extract_data['Distance'] = extract_data['Distance'].apply(lambda x: change_dis(x, avg_dis))
    # 判断是否使用优惠卷
    extract_data['Date'] = extract_data['Date'].fillna(0)
    extract_data['use'] = extract_data['Date']
    extract_data['use'] = extract_data['use'].apply(lambda x: judge(x))
    # 修改折扣率
    extract_data['c_changed_rate'] = extract_data['Discount_rate'].apply(lambda x: change_rate(x))
    label_data['c_changed_rate'] = label_data['Discount_rate'].apply(lambda x: change_rate(x))
    label_data['index'] = label_data.index.tolist()  # 加一列便于区别
    # 计算一个优惠券多少天后使用
    extract_data['use_day'] = 0
    extract_data['use_day'] = extract_data[['use', 'Date_received', 'Date']].apply(lambda x: get_use_day(x), axis=1)
    # 判断优惠券类型
    extract_data['cou_type'] = extract_data['Discount_rate'].apply(lambda x: get_type(x))
    # 满减使用门槛
    extract_data['use_threshold'] = extract_data['Discount_rate'].apply(lambda x: get_max_threshold(x))
    # -----------------------------------------共用变量---------------------------------------------------------
    pre_know_data = label_data.copy()  # 预知结果的数据
    pre_know_data['Date_received'].fillna(0, inplace=True)
    index = pre_know_data[(pre_know_data.Date_received == 0)].index.tolist()
    pre_send_data = pre_know_data.drop(index)  # 标签数据送出的优惠券
    use_cou_data = extract_data.copy()
    index = extract_data[(extract_data.use == 0)].index.tolist()
    use_cou_data = use_cou_data.drop(index)  # 使用过的优惠券
    nuse_cou_data = extract_data.copy()
    index = extract_data[(extract_data.use == 1)].index.tolist()
    nuse_cou_data = nuse_cou_data.drop(index)  # 未使用的优惠券
    send_cou_data = extract_data.copy()
    index = extract_data[(extract_data.Coupon_id == 0)].index.tolist()
    send_cou_data = send_cou_data.drop(index)  # 送出的所有优惠券
    nuse_15days_data = extract_data[(extract_data.label == 0) & (extract_data.use != 1)].copy()
    # -------------------------------------------test----------------------------------------------------------
    # 用户使用距离中位数
    u_med_dis_pivot = pd.pivot_table(extract_data, index=['User_id'], values='Distance', aggfunc=np.median)
    u_med_dis_pivot = DataFrame(u_med_dis_pivot)
    u_med_dis_pivot.columns = ['u_med_dis']
    label_data = pd.merge(label_data, u_med_dis_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 最大时间间隔
    u_max_day_pivot = pd.pivot_table(extract_data, index=['User_id'], values='use_day', aggfunc=max)
    u_max_day_pivot = DataFrame(u_max_day_pivot)
    u_max_day_pivot.columns = ['u_max_day']
    label_data = pd.merge(label_data, u_max_day_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 最小时间间隔
    u_min_day_pivot = pd.pivot_table(extract_data, index=['User_id'], values='use_day', aggfunc=min)
    u_min_day_pivot = DataFrame(u_min_day_pivot)
    u_min_day_pivot.columns = ['u_min_day']
    label_data = pd.merge(label_data, u_min_day_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 平均时间间隔
    u_mean_day_pivot = pd.pivot_table(extract_data, index=['User_id'], values='use_day', aggfunc=np.mean)
    u_mean_day_pivot = DataFrame(u_mean_day_pivot)
    u_mean_day_pivot.columns = ['u_mean_day']
    label_data = pd.merge(label_data, u_mean_day_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 时间间隔中位数
    u_med_day_pivot = pd.pivot_table(extract_data, index=['User_id'], values='use_day', aggfunc=np.median)
    u_med_day_pivot = DataFrame(u_med_day_pivot)
    u_med_day_pivot.columns = ['u_med_day']
    label_data = pd.merge(label_data, u_med_day_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 使用优惠券用户的距离中位数
    u_use_med_dis_pivot = pd.pivot_table(use_cou_data, index=['Coupon_id'], values='Distance', aggfunc=np.median)
    u_use_med_dis_pivot = DataFrame(u_use_med_dis_pivot)
    u_use_med_dis_pivot.columns = ['u_use_med_dis']
    label_data = pd.merge(label_data, u_use_med_dis_pivot, on='Coupon_id', how='left', left_index=True, sort=False)
    # 店铺每种优惠券的数量
    cou_kind_pivot = pd.pivot_table(send_cou_data, index=['Merchant_id', 'Coupon_id'], values='use', aggfunc=len)
    cou_kind_pivot = DataFrame(cou_kind_pivot)
    cou_kind_pivot.columns = ['cou_kind']
    label_data = pd.merge(label_data, cou_kind_pivot, on=['Merchant_id', 'Coupon_id'], how='left', left_index=True,
                          sort=False)
    # --------------预知
    # 用户收到的消费券数量
    l_u_get_sum_pivot = pd.pivot_table(pre_send_data, index=['User_id', 'Coupon_id'], values='Distance', aggfunc=len)
    l_u_get_sum_pivot = DataFrame(l_u_get_sum_pivot)
    l_u_get_sum_pivot.columns = ['l_u_get']
    label_data = pd.merge(label_data, l_u_get_sum_pivot, on=['User_id', 'Coupon_id'], how='left', left_index=True,
                          sort=False)
    # 用户收到的不同消费券数量
    l_u_kind_sum_pivot = pd.pivot_table(pre_send_data, index=['User_id'], values='Coupon_id',
                                        aggfunc=lambda x: len(x.unique()))
    l_u_kind_sum_pivot = DataFrame(l_u_kind_sum_pivot)
    l_u_kind_sum_pivot.columns = ['l_u_kind_sum']
    label_data = pd.merge(label_data, l_u_kind_sum_pivot, on=['User_id'], how='left', left_index=True, sort=False)
    # 用户在不同日期收到的消费券数量
    l_u_get_pivot = pd.pivot_table(pre_send_data, index=['User_id'], values='Coupon_id', aggfunc=len)
    l_u_get_pivot = DataFrame(l_u_get_pivot)
    l_u_get_pivot.columns = ['l_u_get']
    label_data = pd.merge(label_data, l_u_get_pivot, on=['User_id'], how='left', left_index=True, sort=False)
    # 用户在同一天收到的消费券数量
    l_u_dday_get_pivot = pd.pivot_table(pre_send_data, index=['User_id', 'Date_received'], values='Coupon_id',
                                        aggfunc=len)
    l_u_dday_get_pivot = DataFrame(l_u_dday_get_pivot)
    l_u_dday_get_pivot.columns = ['l_u_dday_get']
    label_data = pd.merge(label_data, l_u_dday_get_pivot, on=['User_id', 'Date_received'], how='left', left_index=True,
                          sort=False)
    # 用户收到的不同消费券数量
    l_u_dget_pivot = pd.pivot_table(pre_send_data, index=['User_id'], values='Coupon_id',
                                    aggfunc=lambda x: len(x.unique()))
    l_u_dget_pivot = DataFrame(l_u_dget_pivot)
    l_u_dget_pivot.columns = ['l_u_dget']
    label_data = pd.merge(label_data, l_u_dget_pivot, on=['User_id'], how='left', left_index=True, sort=False)
    # 用户在同一天收到的消费券种类
    l_u_dget_sday_pivot = pd.pivot_table(pre_send_data, index=['User_id', 'Date_received'], values='Coupon_id',
                                         aggfunc=lambda x: len(x.unique()))
    l_u_dget_sday_pivot = DataFrame(l_u_dget_sday_pivot)
    l_u_dget_sday_pivot.columns = ['l_u_dget']
    label_data = pd.merge(label_data, l_u_dget_sday_pivot, on=['User_id', 'Date_received'], how='left', left_index=True,
                          sort=False)
    # 用户在同一天收到的相同消费券数量
    l_u_s_c_d_pivot = pd.pivot_table(pre_send_data, index=['User_id', 'Date_received', 'Coupon_id'],
                                     values='Merchant_id', aggfunc=len)
    l_u_s_c_d_pivot = DataFrame(l_u_s_c_d_pivot)
    l_u_s_c_d_pivot.columns = ['l_u_s_c_d']
    label_data = pd.merge(label_data, l_u_s_c_d_pivot, on=['User_id', 'Date_received', 'Coupon_id'], how='left',
                          left_index=True, sort=False)
    # ---------------------------------------优惠券特征提取-------------------------------------------------------
    # 优惠券使用平均时间间隔
    avg_use_day = pd.pivot_table(extract_data, index=['User_id'], values='use_day', aggfunc=np.mean)
    avg_use_day = DataFrame(avg_use_day)
    avg_use_day.columns = ['avg_use_day']
    label_data = pd.merge(label_data, avg_use_day, on='User_id', how='left')
    # 判断优惠券类型
    extract_data['c_coup_kind'] = extract_data['Discount_rate'].apply(lambda x: get_cou_kind(x))
    # 满减优惠券最低使用
    extract_data['c_min_use'] = extract_data['Discount_rate'].apply(lambda x: get_min_use(x))
    # 优惠券历史出现次数
    c_send_times_pivot = pd.pivot_table(send_cou_data, index=['Coupon_id'], values=['Date_received'],
                                        aggfunc=lambda x: len(x.unique()))
    c_send_times_pivot = DataFrame(c_send_times_pivot)
    c_send_times_pivot.columns = ['c_send_times']
    label_data = pd.merge(label_data, c_send_times_pivot, on='Coupon_id', how='left', left_index=True, sort=False)
    # 优惠券历史使用次数
    c_use_times_pivot = pd.pivot_table(use_cou_data, index=['Coupon_id'], values=['use'], aggfunc=len)
    c_use_times_pivot = DataFrame(c_use_times_pivot)
    c_use_times_pivot.columns = ['c_use_times']
    label_data = pd.merge(label_data, c_use_times_pivot, on='Coupon_id', how='left', left_index=True, sort=False)
    # 优惠券使用率
    ratio10 = c_use_times_pivot['c_use_times'].div(c_send_times_pivot['c_send_times'], axis=0)
    ratio10 = DataFrame(ratio10)
    ratio10.columns = ['ratio10']
    label_data = pd.merge(label_data, ratio10, on='Coupon_id', how='left', left_index=True, sort=False)
    # 优惠券未使用数目
    c_nuse_pivot = pd.pivot_table(nuse_cou_data, index=['Coupon_id'], values=['use'], aggfunc=len)
    c_nuse_pivot = DataFrame(c_nuse_pivot)
    c_nuse_pivot.columns = ['c_nuse']
    label_data = pd.merge(label_data, c_nuse_pivot, on='Coupon_id', how='left', left_index=True, sort=False)
    # 优惠券当天发行多少张
    c_use_day_pivot = pd.pivot_table(send_cou_data, index=['Coupon_id', 'Date_received'], values=['use'], aggfunc=len)
    c_use_day_pivot = DataFrame(c_use_day_pivot)
    c_use_day_pivot.columns = ['c_use_day']
    label_data = pd.merge(label_data, c_use_day_pivot, on=['Coupon_id', 'Date_received'], how='left', left_index=True,
                          sort=False)
    # 不同打折优惠券领取次数
    c_kind_send_pivot = pd.pivot_table(send_cou_data, index=['Coupon_id', 'Discount_rate'], values=['use'], aggfunc=len)
    c_kind_send_pivot = DataFrame(c_kind_send_pivot)
    c_kind_send_pivot.columns = ['c_kind_send']
    label_data = pd.merge(label_data, c_kind_send_pivot, on=['Coupon_id', 'Discount_rate'], how='left', left_index=True,
                          sort=False)
    # 不同打折优惠券使用次数
    c_kind_use_pivot = pd.pivot_table(send_cou_data, index=['Coupon_id', 'Discount_rate'], values=['use'], aggfunc=sum)
    c_kind_use_pivot = DataFrame(c_kind_use_pivot)
    c_kind_use_pivot.columns = ['c_kind_send']
    label_data = pd.merge(label_data, c_kind_use_pivot, on=['Coupon_id', 'Discount_rate'], how='left', left_index=True,
                          sort=False)
    # 不同打折优惠券不使用次数
    c_kind_nuse_pivot = pd.pivot_table(send_cou_data, index=['Coupon_id', 'Discount_rate'], values=['use'],
                                       aggfunc=[len, sum])
    c_kind_nuse_pivot['nuse'] = c_kind_nuse_pivot['len'] - c_kind_nuse_pivot['sum']
    c_kind_nuse_pivot = DataFrame(c_kind_nuse_pivot)
    label_data = pd.merge(label_data, c_kind_nuse_pivot['nuse'], on=['Coupon_id', 'Discount_rate'], how='left',
                          left_index=True, sort=False)
    # 不同打折优惠券使用率
    ratio12 = c_kind_use_pivot['c_kind_send'].div(c_kind_send_pivot['c_kind_send'])
    ratio12 = DataFrame(ratio12)
    ratio12.columns = ['ratio12']
    label_data = pd.merge(label_data, ratio12, on=['Coupon_id', 'Discount_rate'], how='left', left_index=True,
                          sort=False)
    # 优惠券平均核销时间
    c_avg_cost_day_pivot = pd.pivot_table(extract_data, index=['Coupon_id'], values=['use_day'], aggfunc=np.mean)
    c_avg_cost_day_pivot = DataFrame(c_avg_cost_day_pivot)
    c_avg_cost_day_pivot.columns = ['c_avg_cost_day']
    label_data = pd.merge(label_data, c_avg_cost_day_pivot, on=['Coupon_id'], how='left', left_index=True, sort=False)
    # ---------------------------------------用户特征提取-------------------------------------------------------
    # 领券数
    u_cou_sum_pivot = pd.pivot_table(extract_data, index=['User_id'], values=['Coupon_id'], aggfunc=[len])
    u_cou_sum_pivot = DataFrame(u_cou_sum_pivot)
    u_cou_sum_pivot.columns = ['u_cou_sum']
    label_data = pd.merge(label_data, u_cou_sum_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 领券并且消费
    u_used_sum_pivot = pd.pivot_table(extract_data, index=['User_id'], values=['use'], aggfunc=[sum])
    u_used_sum_pivot = DataFrame(u_used_sum_pivot)
    u_used_sum_pivot.columns = ['u_used_sum']
    label_data = pd.merge(label_data, u_used_sum_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 未使用掉优惠券数
    u_nused_sum_pivot = pd.pivot_table(extract_data, index=['User_id'], values=['use'], aggfunc=[len, sum])
    u_nused_sum_pivot = u_nused_sum_pivot['len'] - u_nused_sum_pivot['sum']
    u_nused_sum_pivot = DataFrame(u_nused_sum_pivot)
    u_nused_sum_pivot.columns = ['u_nused_sum']
    label_data = pd.merge(label_data, u_nused_sum_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 领券使用 / 领券不使用
    ratio = u_used_sum_pivot['u_used_sum'].div(u_nused_sum_pivot['u_nused_sum'])
    ratio = DataFrame(ratio)
    ratio.columns = ['ratio']
    label_data = pd.merge(label_data, ratio, on='User_id', how='left', left_index=True, sort=False)
    label_data['ratio'] = label_data['ratio'].apply(lambda x: change(x))
    # 领券并消费数 / 领券数
    temp_pivot = pd.pivot_table(extract_data, index=['User_id'], values=['use'], aggfunc=[sum, len])
    ratio1 = temp_pivot['sum'] / temp_pivot['len']
    ratio1 = DataFrame(ratio1)
    ratio1.columns = ['ratio1']
    label_data = pd.merge(label_data, ratio1, on='User_id', how='left', left_index=True, sort=False)
    # 普通消费
    temp = extract_data[((extract_data.Coupon_id == 0) & (extract_data.Date != 0))]
    u_ord_pivot = pd.pivot_table(temp, index=['User_id'], values='Date', aggfunc=len)
    u_ord_pivot = DataFrame(u_ord_pivot)
    try:
        u_ord_pivot.columns = ['u_ord']
        label_data = pd.merge(label_data, u_ord_pivot, on='User_id', how='left', left_index=True, sort=False)
    except:
        label_data['u_ord'] = 0
    # 总共消费次数
    temp = extract_data[(extract_data.Date != 0)]
    u_all_pivot = pd.pivot_table(temp, index=['User_id'], values='Date', aggfunc=len)
    u_all_pivot = DataFrame(u_all_pivot)
    u_all_pivot.columns = ['u_all']
    label_data = pd.merge(label_data, u_all_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 使用优惠券占比
    u_use_ratio_pivot = u_used_sum_pivot['u_used_sum'].div(u_all_pivot['u_all'])
    u_use_ratio_pivot = DataFrame(u_use_ratio_pivot)
    u_use_ratio_pivot.columns = ['u_use_ratio']
    label_data = pd.merge(label_data, u_use_ratio_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 使用掉的消费券平均折扣
    u_avg_cou_pivot = pd.pivot_table(use_cou_data, index=['User_id'], values=['c_changed_rate'], aggfunc=[sum, len])
    u_avg_cou_pivot = u_avg_cou_pivot['sum'] / u_avg_cou_pivot['len']
    u_avg_cou_pivot = DataFrame(u_avg_cou_pivot)
    u_avg_cou_pivot.columns = ['u_avg_cou']
    label_data = pd.merge(label_data, u_avg_cou_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 使用掉的消费券平均距离
    u_avg_dis_pivot = pd.pivot_table(use_cou_data, index=['User_id'], values=['Distance'], aggfunc=[sum, len])
    u_avg_dis_pivot = u_avg_dis_pivot['sum'] / u_avg_dis_pivot['len']
    u_avg_dis_pivot = DataFrame(u_avg_dis_pivot)
    u_avg_dis_pivot.columns = ['u_avg_dis']
    label_data = pd.merge(label_data, u_avg_dis_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 领取并使用店铺种类
    u_get_use_pivot = pd.pivot_table(use_cou_data, index=['User_id'], values=['Merchant_id'],
                                     aggfunc=lambda x: len(x.unique()))
    u_get_use_pivot = DataFrame(u_get_use_pivot)
    u_get_use_pivot.columns = ['u_get_use']
    label_data = pd.merge(label_data, u_get_use_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 领券店家种类
    u_get_cou_pivot = pd.pivot_table(extract_data, index=['User_id'], values=['Merchant_id'],
                                     aggfunc=lambda x: len(x.unique()))
    u_get_cou_pivot = DataFrame(u_get_cou_pivot)
    u_get_cou_pivot.columns = ['u_get_cou']
    label_data = pd.merge(label_data, u_get_cou_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 在多少不同商家领取并消费优惠券 / 在多少不同商家领取优惠券
    ratio2 = u_get_use_pivot['u_get_use'].div(u_get_cou_pivot['u_get_cou'], axis=0)
    ratio2 = DataFrame(ratio2)
    ratio2.columns = ['ratio2']
    label_data = pd.merge(label_data, ratio2, on='User_id', how='left', left_index=True, sort=False)
    # 最大商家距离
    u_max_dis_pivot = pd.pivot_table(use_cou_data, index=['User_id'], values=['Distance'], aggfunc=[max])
    u_max_dis_pivot = DataFrame(u_max_dis_pivot)
    u_max_dis_pivot.columns = ['u_max_dis']
    label_data = pd.merge(label_data, u_max_dis_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 最小商家距离
    u_min_dis_pivot = pd.pivot_table(use_cou_data, index=['User_id'], values=['Distance'], aggfunc=[min])
    u_min_dis_pivot = DataFrame(u_min_dis_pivot)
    u_min_dis_pivot.columns = ['u_min_dis']
    label_data = pd.merge(label_data, u_min_dis_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 平均店家距离
    u_avg_dis_pivot = pd.pivot_table(use_cou_data, index=['User_id'], values=['Distance'], aggfunc=[np.mean])
    u_avg_dis_pivot = DataFrame(u_avg_dis_pivot)
    u_avg_dis_pivot.columns = ['u_avg_dis']
    label_data = pd.merge(label_data, u_avg_dis_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 最高折扣率
    u_max_count_pivot = pd.pivot_table(use_cou_data, index=['User_id'], values=['c_changed_rate'], aggfunc=[max])
    u_max_count_pivot = DataFrame(u_max_count_pivot)
    u_max_count_pivot.columns = ['u_max_count']
    label_data = pd.merge(label_data, u_max_count_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 最低折扣率
    u_min_count_pivot = pd.pivot_table(use_cou_data, index=['User_id'], values=['c_changed_rate'], aggfunc=[min])
    u_min_count_pivot = DataFrame(u_min_count_pivot)
    u_min_count_pivot.columns = ['u_min_count']
    label_data = pd.merge(label_data, u_min_count_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 平均折扣率
    u_avg_count_pivot = pd.pivot_table(use_cou_data, index=['User_id'], values=['c_changed_rate'], aggfunc=[np.mean])
    u_avg_count_pivot = DataFrame(u_avg_count_pivot)
    u_avg_count_pivot.columns = ['u_avg_count']
    label_data = pd.merge(label_data, u_avg_count_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 15天普通消费平均一天几次
    ord_data = extract_data[(extract_data.Date != 0) & (extract_data.Coupon_id == 0) & (extract_data.label == 1)]
    u_avg_ord15_pivot = pd.pivot_table(ord_data, index=['User_id'], values=['use'], aggfunc=len)
    u_avg_ord15_pivot = DataFrame(u_avg_ord15_pivot)
    try:
        u_avg_ord15_pivot.columns = ['u_avg_ord']
        u_avg_ord15_pivot['u_avg_ord'] /= 15
        label_data = pd.merge(label_data, u_avg_ord15_pivot, on='User_id', how='left', left_index=True, sort=False)
    except:
        label_data['u_avg_ord'] = 0
    # 15天优惠券消费平均一天几次
    cou15_data = extract_data[(extract_data.Date != 0) & (extract_data.Coupon_id != 0) & (extract_data.label == 1)]
    u_avg_use15_pivot = pd.pivot_table(cou15_data, index=['User_id'], values=['use'], aggfunc=len)
    u_avg_use15_pivot = DataFrame(u_avg_use15_pivot)
    u_avg_use15_pivot.columns = ['u_avg_ord']
    u_avg_use15_pivot['u_avg_ord'] /= 15
    label_data = pd.merge(label_data, u_avg_use15_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 15天内使用优惠券的次数
    u_use_pivot = pd.pivot_table(cou15_data, index=['User_id'], values=['use'], aggfunc=len)
    u_use_pivot = DataFrame(u_use_pivot)
    u_use_pivot.columns = ['u_use']
    label_data = pd.merge(label_data, u_use_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 15天使用用占总使用多少
    all_use = pd.pivot_table(use_cou_data, index=['User_id'], values=['use'], aggfunc=len)
    all_use = DataFrame(all_use)
    all_use.columns = ['all_use']
    ratio13 = u_use_pivot['u_use'].div(all_use['all_use'])
    ratio13 = DataFrame(ratio13)
    ratio13.columns = ['ratio13']
    label_data = pd.merge(label_data, ratio13, on='User_id', how='left', left_index=True, sort=False)
    # 15天使用占未使用多少
    all_nuse = pd.pivot_table(nuse_15days_data, index=['User_id'], values=['use'], aggfunc=len)
    all_nuse = DataFrame(all_nuse)
    all_nuse.columns = ['all_nuse']
    ratio14 = u_use_pivot['u_use'].div(all_nuse['all_nuse'])
    ratio14 = DataFrame(ratio14)
    ratio14.columns = ['ratio14']
    label_data = pd.merge(label_data, ratio14, on='User_id', how='left', left_index=True, sort=False)
    # 15天使用占总领取
    all_get = pd.pivot_table(send_cou_data, index=['User_id'], values=['use'], aggfunc=len)
    all_get = DataFrame(all_get)
    all_get.columns = ['all_get']
    ratio15 = u_use_pivot['u_use'].div(all_get['all_get'])
    ratio15 = DataFrame(ratio15)
    ratio15.columns = ['ratio15']
    label_data = pd.merge(label_data, ratio15, on='User_id', how='left', left_index=True, sort=False)
    # 用户平均一种优惠券使用多少张
    u_get_cou_kind = pd.pivot_table(send_cou_data, index=['User_id'], values=['Coupon_id'],
                                    aggfunc=lambda x: len(x.unique()))
    u_get_cou_kind = DataFrame(u_get_cou_kind)
    u_get_cou_kind.columns = ['u_get_cou_kind']
    u_avg_use_kind_pivot = u_cou_sum_pivot['u_cou_sum'].div(u_get_cou_kind['u_get_cou_kind'])
    u_avg_use_kind_pivot = DataFrame(u_avg_use_kind_pivot)
    u_avg_use_kind_pivot.columns = ['u_avg_use_kind']
    label_data = pd.merge(label_data, u_avg_use_kind_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 消费平均时间间隔
    shopping_data = extract_data[(extract_data.Date != 0)].copy()
    shopping_data['Date'] = pd.to_datetime(shopping_data['Date'])
    avg_days_pivot = pd.pivot_table(shopping_data, index=['User_id'], values=['Date'],
                                    aggfunc=lambda x: get_avg_days(x))
    avg_days_pivot = DataFrame(avg_days_pivot)
    avg_days_pivot.columns = ['avg_days']
    label_data = pd.merge(label_data, avg_days_pivot, on=['User_id'], how='left', left_index=True, sort=False)
    # ---------------------------------------店家特征提取--------------------------------------------------------
    # 店家送券数量
    send_data = extract_data[(extract_data.Coupon_id != -1)]  # 送出券的店家
    s_cou_get_pivot = pd.pivot_table(send_data, index='Merchant_id', values=['Coupon_id'], aggfunc=len)
    s_cou_get_pivot = DataFrame(s_cou_get_pivot)
    s_cou_get_pivot.columns = ['s_cou_get']
    label_data = pd.merge(label_data, s_cou_get_pivot, on='Merchant_id', how='left', left_index=True, sort=False)
    # 店家送券使用数量
    s_cou_use_pivot = pd.pivot_table(use_cou_data, index='Merchant_id', values=['Coupon_id'], aggfunc=len)
    s_cou_use_pivot = DataFrame(s_cou_use_pivot)
    s_cou_use_pivot.columns = ['s_cou_use']
    label_data = pd.merge(label_data, s_cou_use_pivot, on='Merchant_id', how='left', left_index=True, sort=False)
    # 店家送券不使用
    s_cou_nuse_pivot = pd.pivot_table(nuse_cou_data, index='Merchant_id', values=['Coupon_id'], aggfunc=len)
    s_cou_nuse_pivot = DataFrame(s_cou_nuse_pivot)
    s_cou_nuse_pivot.columns = ['s_cou_nuse']
    label_data = pd.merge(label_data, s_cou_nuse_pivot, on='Merchant_id', how='left', left_index=True, sort=False)
    # 店家优惠券使用率
    ratio3 = s_cou_use_pivot['s_cou_use'].div(s_cou_get_pivot['s_cou_get'], axis=0)
    ratio3 = DataFrame(ratio3)
    ratio3.columns = ['ratio3']
    label_data = pd.merge(label_data, ratio3, on='Merchant_id', how='left', left_index=True, sort=False)
    # 店家优惠券最大折扣率
    s_max_discount_pivot = pd.pivot_table(extract_data, index='Merchant_id', values=['c_changed_rate'], aggfunc=max)
    s_max_discount_pivot = DataFrame(s_max_discount_pivot)
    s_max_discount_pivot.columns = ['s_max_discount']
    label_data = pd.merge(label_data, s_max_discount_pivot, on='Merchant_id', how='left', left_index=True, sort=False)
    # 店家优惠券最小折扣率
    s_min_discount_pivot = pd.pivot_table(extract_data, index='Merchant_id', values=['c_changed_rate'], aggfunc=min)
    s_min_discount_pivot = DataFrame(s_min_discount_pivot)
    s_min_discount_pivot.columns = ['s_min_discount']
    label_data = pd.merge(label_data, s_min_discount_pivot, on='Merchant_id', how='left', left_index=True, sort=False)
    # 店家优惠券平均折扣率
    s_avg_discount_pivot = pd.pivot_table(extract_data, index='Merchant_id', values=['c_changed_rate'], aggfunc=np.mean)
    s_avg_discount_pivot = DataFrame(s_avg_discount_pivot)
    s_avg_discount_pivot.columns = ['s_avg_discount']
    label_data = pd.merge(label_data, s_avg_discount_pivot, on='Merchant_id', how='left', left_index=True, sort=False)
    # 使用优惠券不同用户数量
    s_use_cus_pivot = pd.pivot_table(use_cou_data, index=['Merchant_id'], values=['User_id'],
                                     aggfunc=lambda x: len(x.unique()))
    s_use_cus_pivot = DataFrame(s_use_cus_pivot)
    s_use_cus_pivot.columns = ['s_use_cus']
    label_data = pd.merge(label_data, s_use_cus_pivot, on='Merchant_id', how='left', left_index=True, sort=False)
    # 店家优惠券一个用户用几张
    s_cus_avguse_pivot = s_cou_use_pivot['s_cou_use'].div(s_use_cus_pivot['s_use_cus'], axis=0)
    s_cus_avguse_pivot = DataFrame(s_cus_avguse_pivot)
    s_cus_avguse_pivot.columns = ['s_cus_avguse']
    label_data = pd.merge(label_data, s_cus_avguse_pivot, on='Merchant_id', how='left', left_index=True, sort=False)
    # 店家优惠券种类
    s_cou_kind_pivot = pd.pivot_table(extract_data, index=['Merchant_id'], values=['Coupon_id'],
                                      aggfunc=lambda x: len(x.unique()))
    s_cou_kind_pivot = DataFrame(s_cou_kind_pivot)
    s_cou_kind_pivot.columns = ['s_cou_kind']
    label_data = pd.merge(label_data, s_cou_kind_pivot, on='Merchant_id', how='left', left_index=True, sort=False)
    # 店家被使用的优惠券种类
    s_diff_cou_pivot = pd.pivot_table(use_cou_data, index=['Merchant_id'], values=['Coupon_id'],
                                      aggfunc=lambda x: len(x.unique()))
    s_diff_cou_pivot = DataFrame(s_diff_cou_pivot)
    s_diff_cou_pivot.columns = ['s_diff_cou']
    label_data = pd.merge(label_data, s_diff_cou_pivot, on='Merchant_id', how='left', left_index=True, sort=False)
    # 使用种类 / 发放使用种类
    ratio4 = s_diff_cou_pivot['s_diff_cou'].div(s_cou_kind_pivot['s_cou_kind'], axis=0)
    ratio4 = DataFrame(ratio4)
    ratio4.columns = ['ratio4']
    label_data = pd.merge(label_data, ratio4, on='Merchant_id', how='left', left_index=True, sort=False)
    # 店家使用优惠券最大距离
    s_max_dis_pivot = pd.pivot_table(extract_data, index=['Merchant_id'], values=['Distance'], aggfunc=max)
    s_max_dis_pivot = DataFrame(s_max_dis_pivot)
    s_max_dis_pivot.columns = ['s_max_dis']
    label_data = pd.merge(label_data, s_max_dis_pivot, on='Merchant_id', how='left', left_index=True, sort=False)
    # 店家使用优惠券最小距离
    s_min_dis_pivot = pd.pivot_table(extract_data, index=['Merchant_id'], values=['Distance'], aggfunc=min)
    s_min_dis_pivot = DataFrame(s_min_dis_pivot)
    s_min_dis_pivot.columns = ['s_min_dis']
    label_data = pd.merge(label_data, s_min_dis_pivot, on='Merchant_id', how='left', left_index=True, sort=False)
    # 店家使用优惠券平均距离
    s_mean_dis_pivot = pd.pivot_table(extract_data, index=['Merchant_id'], values=['Distance'], aggfunc=np.mean)
    s_mean_dis_pivot = DataFrame(s_mean_dis_pivot)
    s_mean_dis_pivot.columns = ['s_avg_dis']
    label_data = pd.merge(label_data, s_mean_dis_pivot, on='Merchant_id', how='left', left_index=True, sort=False)
    # 店家消费次数
    shop_data = extract_data[(extract_data.Date != 0)]
    s_shop_pivot = pd.pivot_table(shop_data, index=['Merchant_id'], values=['use'], aggfunc=len)
    s_shop_pivot = DataFrame(s_shop_pivot)
    s_shop_pivot.columns = ['s_shop']
    label_data = pd.merge(label_data, s_shop_pivot, on='Merchant_id', how='left', left_index=True, sort=False)
    # 店家正常消费次数
    s_ord_shop_pivot = pd.pivot_table(ord_data, index=['Merchant_id'], values=['use'], aggfunc=len)
    s_ord_shop_pivot = DataFrame(s_ord_shop_pivot)
    try:
        s_ord_shop_pivot.columns = ['s_ord_shop']
        label_data = pd.merge(label_data, s_ord_shop_pivot, on='Merchant_id', how='left', left_index=True, sort=False)
    except:
        label_data['s_ord_shop'] = 0
    # 店家当天优惠券领取次数
    s_send_day_pivot = pd.pivot_table(send_cou_data, index=['Merchant_id', 'Date_received'], values=['use'],
                                      aggfunc=len)
    s_send_day_pivot = DataFrame(s_send_day_pivot)
    s_send_day_pivot.columns = ['s_send_day']
    label_data = pd.merge(label_data, s_send_day_pivot, on=['Merchant_id', 'Date_received'], how='left',
                          left_index=True, sort=False)
    # 店家当天优惠券领取人数
    s_send_peo_pivot = pd.pivot_table(send_cou_data, index=['Merchant_id', 'Date_received'], values=['User_id'],
                                      aggfunc=lambda x: len(x.unique()))
    s_send_peo_pivot = DataFrame(s_send_peo_pivot)
    s_send_peo_pivot.columns = ['s_send_peo']
    label_data = pd.merge(label_data, s_send_peo_pivot, on=['Merchant_id', 'Date_received'], how='left',
                          left_index=True, sort=False)
    # 店家优惠券的平均使用时间
    s_use_day_pivot = pd.pivot_table(send_cou_data, index=['Merchant_id'], values=['use_day'], aggfunc=np.mean)
    s_use_day_pivot = DataFrame(s_use_day_pivot)
    s_use_day_pivot.columns = ['s_use_day']
    label_data = pd.merge(label_data, s_use_day_pivot, on='Merchant_id', how='left', left_index=True, sort=False)
    # ---------------------------------------店家-用户特征提取-------------------------------------------------------
    # 用户领取店家优惠券次数
    u_s_get_pivot = pd.pivot_table(extract_data, index=['User_id', 'Merchant_id'], values=['use'], aggfunc=len)
    u_s_get_pivot = DataFrame(u_s_get_pivot)
    u_s_get_pivot.columns = ['u_s_get']
    label_data = pd.merge(label_data, u_s_get_pivot, on=['User_id', 'Merchant_id'], how='left', left_index=True,
                          sort=False)
    # 用户领取不使用次数
    u_s_nuse_pivot = pd.pivot_table(nuse_cou_data, index=['User_id', 'Merchant_id'], values=['use'], aggfunc=len)
    u_s_nuse_pivot = DataFrame(u_s_nuse_pivot)
    u_s_nuse_pivot.columns = ['u_s_nuse']
    label_data = pd.merge(label_data, u_s_nuse_pivot, on=['User_id', 'Merchant_id'], how='left', left_index=True,
                          sort=False)
    # 用户领取并使用
    u_s_use_pivot = pd.pivot_table(use_cou_data, index=['User_id', 'Merchant_id'], values=['use'], aggfunc=len)
    u_s_use_pivot = DataFrame(u_s_use_pivot)
    u_s_use_pivot.columns = ['u_s_use']
    label_data = pd.merge(label_data, u_s_use_pivot, on=['User_id', 'Merchant_id'], how='left', left_index=True,
                          sort=False)
    # 领取使用率
    ratio5 = u_s_use_pivot['u_s_use'].div(u_s_get_pivot['u_s_get'], axis=0)
    ratio5 = DataFrame(ratio5)
    ratio5.columns = ['ratio5']
    label_data = pd.merge(label_data, ratio5, on=['User_id', 'Merchant_id'], how='left', left_index=True, sort=False)
    # 单个商家不使用占总的不使用比率
    all_nuse_pivot = pd.pivot_table(nuse_cou_data, index=['User_id'], values=['use'], aggfunc=len)
    all_nuse_pivot = DataFrame(all_nuse_pivot)
    all_nuse_pivot.columns = ['all_nuse']
    ratio6 = u_s_nuse_pivot['u_s_nuse'].div(all_nuse_pivot['all_nuse'], axis=0)
    ratio6 = DataFrame(ratio6)
    ratio6.columns = ['ratio6']
    label_data = pd.merge(label_data, ratio6, on=['User_id', 'Merchant_id'], how='left', left_index=True, sort=False)
    # 单个商家使用占总的使用比率
    all_use_pivot = pd.pivot_table(use_cou_data, index=['User_id'], values=['use'], aggfunc=len)
    all_use_pivot = DataFrame(all_use_pivot)
    all_use_pivot.columns = ['all_use']
    ratio7 = u_s_use_pivot['u_s_use'].div(all_use_pivot['all_use'], axis=0)
    ratio7 = DataFrame(ratio7)
    ratio7.columns = ['ratio7']
    label_data = pd.merge(label_data, ratio7, on=['User_id', 'Merchant_id'], how='left', left_index=True, sort=False)
    # 用户对店家使用占店家总使用比率
    ratio8 = u_s_use_pivot['u_s_use'].div(s_cou_use_pivot['s_cou_use'], axis=0)
    ratio8 = DataFrame(ratio8)
    ratio8.columns = ['ratio8']
    label_data = pd.merge(label_data, ratio8, on=['User_id', 'Merchant_id'], how='left', left_index=True, sort=False)
    # 用户对店家不使用占店家总不使用比率
    ratio9 = u_s_nuse_pivot['u_s_nuse'].div(s_cou_nuse_pivot['s_cou_nuse'], axis=0)
    ratio9 = DataFrame(ratio9)
    ratio9.columns = ['ratio9']
    label_data = pd.merge(label_data, ratio9, on=['User_id', 'Merchant_id'], how='left', left_index=True, sort=False)
    # 用户在此店消费几次
    u_s_shop_time_pivot = pd.pivot_table(shop_data, index=['User_id', 'Merchant_id'], values=['use'], aggfunc=len)
    u_s_shop_time_pivot = DataFrame(u_s_shop_time_pivot)
    u_s_shop_time_pivot.columns = ['u_s_shop_time']
    label_data = pd.merge(label_data, u_s_shop_time_pivot, on=['User_id', 'Merchant_id'], how='left', left_index=True,
                          sort=False)
    # 用户在此店普通消费几次
    u_s_ord_time_pivot = pd.pivot_table(extract_data, index=['User_id', 'Merchant_id'], values=['use'], aggfunc=len)
    u_s_ord_time_pivot = DataFrame(u_s_ord_time_pivot)
    u_s_ord_time_pivot.columns = ['u_s_ord_time']
    label_data = pd.merge(label_data, u_s_ord_time_pivot, on=['User_id', 'Merchant_id'], how='left', left_index=True,
                          sort=False)
    # 用户在同一天在此店领取优惠券数量
    u_s_c_same_day_pivot = pd.pivot_table(send_cou_data, index=['User_id', 'Merchant_id', 'Date_received'],
                                          values=['use'], aggfunc=len)
    u_s_c_same_day_pivot = DataFrame(u_s_c_same_day_pivot)
    u_s_c_same_day_pivot.columns = ['u_s_c_same_day']
    label_data = pd.merge(label_data, u_s_c_same_day_pivot, on=['User_id', 'Merchant_id', 'Date_received'], how='left',
                          left_index=True, sort=False)
    # 用户领取优惠券的不同店家数量
    u_c_diff_s_pivot = pd.pivot_table(send_cou_data, index=['User_id'], values=['Merchant_id'],
                                      aggfunc=lambda x: len(x.unique()))
    u_c_diff_s_pivot = DataFrame(u_c_diff_s_pivot)
    u_c_diff_s_pivot.columns = ['u_c_diff_s']
    label_data = pd.merge(label_data, u_c_diff_s_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 用户使用优惠券的不同店家数量
    u_s_diff_use_pivot = pd.pivot_table(use_cou_data, index=['User_id'], values=['Merchant_id'],
                                        aggfunc=lambda x: len(x.unique()))
    u_s_diff_use_pivot = DataFrame(u_s_diff_use_pivot)
    u_s_diff_use_pivot.columns = ['u_s_diff_use']
    label_data = pd.merge(label_data, u_s_diff_use_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 使用的店家与领取的所有店家比值
    ratio16 = u_s_diff_use_pivot['u_s_diff_use'].div(u_c_diff_s_pivot['u_c_diff_s'])
    ratio16 = DataFrame(ratio16)
    ratio16.columns = ['ratio16']
    label_data = pd.merge(label_data, ratio16, on='User_id', how='left', left_index=True, sort=False)
    # 一个店使用多少张优惠券
    s_c_avg_use = u_used_sum_pivot['u_used_sum'].div(u_s_diff_use_pivot['u_s_diff_use'])
    s_c_avg_use = DataFrame(s_c_avg_use)
    s_c_avg_use.columns = ['s_c_avg_use']
    label_data = pd.merge(label_data, s_c_avg_use, on='User_id', how='left', left_index=True, sort=False)
    # -------------------------------------优惠券-用户特征提取--------------------------------------------------------
    # 用户领取该优惠券几次
    u_c_get_pivot = pd.pivot_table(send_cou_data, index=['User_id', 'Coupon_id'], values=['Date_received'], aggfunc=len)
    u_c_get_pivot = DataFrame(u_c_get_pivot)
    u_c_get_pivot.columns = ['u_c_get']
    label_data = pd.merge(label_data, u_c_get_pivot, on=['User_id', 'Coupon_id'], how='left', left_index=True,
                          sort=False)
    # 用户使用该优惠券几次
    u_c_use_pivot = pd.pivot_table(use_cou_data, index=['User_id', 'Coupon_id'], values=['use'], aggfunc=len)
    u_c_use_pivot = DataFrame(u_c_use_pivot)
    u_c_use_pivot.columns = ['u_c_use']
    label_data = pd.merge(label_data, u_c_use_pivot, on=['User_id', 'Coupon_id'], how='left', left_index=True,
                          sort=False)
    # 用户使用率
    ratio11 = u_c_use_pivot['u_c_use'].div(u_c_get_pivot['u_c_get'], axis=0)
    ratio11 = DataFrame(ratio11)
    ratio11.columns = ['ratio11']
    label_data = pd.merge(label_data, ratio11, on=['User_id', 'Coupon_id'], how='left', left_index=True, sort=False)
    # 用户不同折扣优惠券使用次数
    u_c_diff_use_time_pivot = pd.pivot_table(use_cou_data, index=['User_id', 'Discount_rate'], values=['use'],
                                             aggfunc=len)
    u_c_diff_use_time_pivot = DataFrame(u_c_diff_use_time_pivot)
    u_c_diff_use_time_pivot.columns = ['u_c_diff_use_time1']
    label_data = pd.merge(label_data, u_c_diff_use_time_pivot, on=['User_id', 'Discount_rate'], how='left',
                          left_index=True, sort=False)

    u_c_diff_use_time_pivot = pd.pivot_table(use_cou_data, index=['User_id', 'c_changed_rate'], values=['use'],
                                             aggfunc=len)
    u_c_diff_use_time_pivot = DataFrame(u_c_diff_use_time_pivot)
    u_c_diff_use_time_pivot.columns = ['u_c_diff_use_time2']
    label_data = pd.merge(label_data, u_c_diff_use_time_pivot, on=['User_id', 'c_changed_rate'], how='left',
                          left_index=True, sort=False)
    # 用户不同优惠券不使用次数
    u_c_diff_nuse_time_pivot = pd.pivot_table(nuse_cou_data, index=['User_id', 'Discount_rate'], values=['use'],
                                              aggfunc=len)
    u_c_diff_nuse_time_pivot = DataFrame(u_c_diff_nuse_time_pivot)
    u_c_diff_nuse_time_pivot.columns = ['u_c_diff_nuse_time1']
    label_data = pd.merge(label_data, u_c_diff_nuse_time_pivot, on=['User_id', 'Discount_rate'], how='left',
                          left_index=True, sort=False)

    u_c_diff_nuse_time_pivot = pd.pivot_table(nuse_cou_data, index=['User_id', 'c_changed_rate'], values=['use'],
                                              aggfunc=len)
    u_c_diff_nuse_time_pivot = DataFrame(u_c_diff_nuse_time_pivot)
    u_c_diff_nuse_time_pivot.columns = ['u_c_diff_nuse_time2']
    label_data = pd.merge(label_data, u_c_diff_nuse_time_pivot, on=['User_id', 'c_changed_rate'], how='left',
                          left_index=True, sort=False)
    # ------------------------------------------其他特征提取-----------------------------------------------------------
    # 用户一天领取多少优惠券
    u_c_get_same_sum_pivot = pd.pivot_table(extract_data, index=['User_id', 'Date_received'], values=['Coupon_id'],
                                            aggfunc=len)
    u_c_get_same_sum_pivot = DataFrame(u_c_get_same_sum_pivot)
    u_c_get_same_sum_pivot.columns = ['u_c_get_same_sum']
    label_data = pd.merge(label_data, u_c_get_same_sum_pivot, on=['User_id', 'Date_received'], how='left',
                          left_index=True, sort=False)
    # 用户同一天领取同一优惠券次数
    u_c_get_same_day_pivot = pd.pivot_table(extract_data, index=['User_id', 'Coupon_id', 'Date_received'],
                                            values=['use'], aggfunc=len)
    u_c_get_same_day_pivot = DataFrame(u_c_get_same_day_pivot)
    u_c_get_same_day_pivot.columns = ['u_c_get_same_day']
    label_data = pd.merge(label_data, u_c_get_same_day_pivot, on=['User_id', 'Coupon_id', 'Date_received'], how='left',
                          left_index=True, sort=False)
    # 用户领取相同优惠券次数
    u_c_get_same_times_pivot = pd.pivot_table(extract_data, index=['User_id', 'Coupon_id'], values=['use'], aggfunc=len)
    u_c_get_same_times_pivot = DataFrame(u_c_get_same_times_pivot)
    u_c_get_same_times_pivot.columns = ['u_c_get_same_times']
    label_data = pd.merge(label_data, u_c_get_same_times_pivot, on=['User_id', 'Coupon_id'], how='left',
                          left_index=True, sort=False)
    # ----------------------------------------预知特征提取-----------------------------------------------------------
    # 用户领取所有优惠券数量
    l_cou_sum_pivot = pd.pivot_table(label_data, index=['User_id'], values=['Coupon_id'], aggfunc=len)
    l_cou_sum_pivot = DataFrame(l_cou_sum_pivot)
    l_cou_sum_pivot.columns = ['l_cou_sum']
    label_data = pd.merge(label_data, l_cou_sum_pivot, on=['User_id'], how='left', left_index=True, sort=False)
    # 用户领取相同优惠券数量
    l_c_get_same_times_pivot = pd.pivot_table(label_data, index=['User_id', 'Coupon_id'], values=['index'], aggfunc=len)
    l_c_get_same_times_pivot = DataFrame(l_c_get_same_times_pivot)
    l_c_get_same_times_pivot.columns = ['l_c_get_same_times']
    label_data = pd.merge(label_data, l_c_get_same_times_pivot, on=['User_id', 'Coupon_id'], how='left',
                          left_index=True, sort=False)
    # 用户领取相同店家优惠券数量
    l_same_s_pivot = pd.pivot_table(label_data, index=['User_id', 'Merchant_id'], values=['index'], aggfunc=len)
    l_same_s_pivot = DataFrame(l_same_s_pivot)
    l_same_s_pivot.columns = ['l_same_s']
    label_data = pd.merge(label_data, l_same_s_pivot, on=['User_id', 'Merchant_id'], how='left', left_index=True,
                          sort=False)
    # 用户领取不同店家数量
    l_s_sum_pivot = pd.pivot_table(label_data, index=['User_id'], values=['Merchant_id'],
                                   aggfunc=lambda x: len(x.unique()))
    l_s_sum_pivot = DataFrame(l_s_sum_pivot)
    l_s_sum_pivot.columns = ['l_s_sum']
    label_data = pd.merge(label_data, l_s_sum_pivot, on=['User_id'], how='left', left_index=True, sort=False)
    # 用户同一天领取的优惠券数量
    l_sum_c_day_pivot = pd.pivot_table(label_data, index=['User_id', 'Date_received'], values=['index'], aggfunc=len)
    l_sum_c_day_pivot = DataFrame(l_sum_c_day_pivot)
    l_sum_c_day_pivot.columns = ['l_same_c_day']
    label_data = pd.merge(label_data, l_sum_c_day_pivot, on=['User_id', 'Date_received'], how='left', left_index=True,
                          sort=False)
    # 用户同一天领取相同优惠券数量
    label_data['temp'] = 1
    l_same_c_day_pivot = pd.pivot_table(label_data, index=['User_id', 'Date_received', 'Coupon_id'], values=['temp'],
                                        aggfunc=len)
    l_same_c_day_pivot = DataFrame(l_same_c_day_pivot)
    l_same_c_day_pivot.columns = ['l_same_c_day']
    label_data = pd.merge(label_data, l_same_c_day_pivot, on=['User_id', 'Date_received', 'Coupon_id'], how='left',
                          left_index=True, sort=False)
    label_data.drop(['temp'], inplace=True, axis=1)
    # 用户领取的优惠券种类
    l_c_kind_pivot = pd.pivot_table(label_data, index=['User_id'], values=['Coupon_id'],
                                    aggfunc=lambda x: len(x.unique()))
    l_c_kind_pivot = DataFrame(l_c_kind_pivot)
    l_c_kind_pivot.columns = ['l_c_kind']
    label_data = pd.merge(label_data, l_c_kind_pivot, on=['User_id'], how='left', left_index=True, sort=False)
    # 商家被领取的优惠券总量
    l_s_c_kind_pivot = pd.pivot_table(pre_send_data, index=['Merchant_id'], values=['Coupon_id'],
                                      aggfunc=lambda x: len(x.unique()))
    l_s_c_kind_pivot = DataFrame(l_s_c_kind_pivot)
    l_s_c_kind_pivot.columns = ['l_s_c_kind']
    label_data = pd.merge(label_data, l_s_c_kind_pivot, on=['Merchant_id'], how='left', left_index=True, sort=False)
    # 商家被领取的相同优惠券数量
    l_s_c_same_pivot = pd.pivot_table(pre_send_data, index=['Merchant_id', 'Coupon_id'], values=['User_id'],
                                      aggfunc=len)
    l_s_c_same_pivot = DataFrame(l_s_c_same_pivot)
    l_s_c_same_pivot.columns = ['l_s_c_same']
    label_data = pd.merge(label_data, l_s_c_same_pivot, on=['Merchant_id', 'Coupon_id'], how='left', left_index=True,
                          sort=False)
    # 商家优惠券被多少种不同用户领取
    l_s_c_u_kind_pivot = pd.pivot_table(pre_send_data, index=['Merchant_id'], values=['User_id'],
                                        aggfunc=lambda x: len(x.unique()))
    l_s_c_u_kind_pivot = DataFrame(l_s_c_u_kind_pivot)
    l_s_c_u_kind_pivot.columns = ['l_s_c_u_kind']
    label_data = pd.merge(label_data, l_s_c_u_kind_pivot, on=['Merchant_id'], how='left', left_index=True, sort=False)
    # 商家的优惠券种类
    l_s_c_kind_pivot = pd.pivot_table(label_data, index=['Merchant_id'], values=['Coupon_id'],
                                      aggfunc=lambda x: len(x.unique()))
    l_s_c_kind_pivot = DataFrame(l_s_c_kind_pivot)
    l_s_c_kind_pivot.columns = ['s_c_kind']
    label_data = pd.merge(label_data, l_s_c_kind_pivot, on=['Merchant_id'], how='left', left_index=True, sort=False)
    label_data.drop('index', axis=1, inplace=True)
    label_data.index = data_index
    # 用户领取该优惠券占总领取比值
    l_cou_same_pivot = pd.pivot_table(label_data, index=['User_id', 'Coupon_id'], values=['Merchant_id'], aggfunc=len)
    l_cou_same_pivot = DataFrame(l_cou_same_pivot)
    l_cou_same_pivot.columns = ['l_cou_same']
    ratio = l_cou_same_pivot['l_cou_same'].div(l_cou_sum_pivot['l_cou_sum'], axis=0)
    ratio = DataFrame(ratio)
    ratio.columns = ['ratio']
    label_data = pd.merge(label_data, ratio, on=['User_id', 'Coupon_id'], how='left', left_index=True, sort=False)
    # 用户该店铺领取数量占总店铺比值
    l_cou_same_pivot = pd.pivot_table(label_data, index=['User_id', 'Merchant_id'], values=['Coupon_id'], aggfunc=len)
    l_cou_same_pivot = DataFrame(l_cou_same_pivot)
    l_cou_same_pivot.columns = ['l_cou_same']
    ratio = l_cou_same_pivot['l_cou_same'].div(l_cou_sum_pivot['l_cou_sum'], axis=0)
    ratio = DataFrame(ratio)
    ratio.columns = ['ratio']
    label_data = pd.merge(label_data, ratio, on=['User_id', 'Merchant_id'], how='left', left_index=True, sort=False)
    return label_data


def get_on_feature(dataset, online_data, offline_data):
    data_index = dataset.index
    online_data = online_data.fillna(0)
    click_data = online_data.copy()
    click_data = click_data[((click_data.Action == 0) | (click_data.Action == 2))]  # 点击操作数据
    buy_data = online_data.copy()
    buy_data = buy_data[(buy_data.Action == 1)]  # 购买操作数据
    get_data = online_data.copy()
    get_data = get_data[(get_data.Action == 2)]  # 领取优惠券数据
    shopping_data = online_data.copy()
    index = online_data[online_data.Date == -1].index.tolist()
    shopping_data = shopping_data.drop(index)  # 线上消费数据
    use_cou_data = online_data[(online_data.Coupon_id != 0) & (online_data.Date != 0)].copy()
    # 线上点次数
    on_click_pivot = pd.pivot_table(click_data, index=['User_id'], values=['Action'], aggfunc=len)
    on_click_pivot = DataFrame(on_click_pivot)
    on_click_pivot.columns = ['on_click']
    dataset = pd.merge(dataset, on_click_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 线上操作次数
    on_action_pivot = pd.pivot_table(online_data, index=['User_id'], values=['Action'], aggfunc=len)
    on_action_pivot = DataFrame(on_action_pivot)
    on_action_pivot.columns = ['on_action']
    dataset = pd.merge(dataset, on_action_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 点击率
    on_click_ratio_pivot = on_click_pivot['on_click'].div(on_action_pivot['on_action'], axis=0)
    on_click_ratio_pivot = DataFrame(on_click_ratio_pivot)
    on_click_ratio_pivot.columns = ['on_click_ratio']
    dataset = pd.merge(dataset, on_click_ratio_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 线上购买次数
    on_buy_pivot = pd.pivot_table(buy_data, index=['User_id'], values=['Action'], aggfunc=len)
    on_buy_pivot = DataFrame(on_buy_pivot)
    on_buy_pivot.columns = ['on_buy']
    dataset = pd.merge(dataset, on_buy_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 线上购买率
    on_buy_rate_pivot = on_buy_pivot['on_buy'].div(on_click_pivot['on_click'], axis=0)
    on_buy_rate_pivot = DataFrame(on_buy_rate_pivot)
    on_buy_rate_pivot.columns = ['on_buy_rate']
    dataset = pd.merge(dataset, on_buy_rate_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 线上领取次数
    on_get_pivot = pd.pivot_table(get_data, index=['User_id'], values=['Action'], aggfunc=len)
    on_get_pivot = DataFrame(on_get_pivot)
    on_get_pivot.columns = ['on_get']
    dataset = pd.merge(dataset, on_get_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 线上领取率
    on_get_ratio_pivot = on_get_pivot['on_get'].div(on_action_pivot['on_action'], axis=0)
    on_get_ratio_pivot = DataFrame(on_get_ratio_pivot)
    on_get_ratio_pivot.columns = ['on_get_ratio']
    dataset = pd.merge(dataset, on_get_ratio_pivot, on=['User_id'], how='left', left_index=True, sort=False)
    # 线上消费次数
    on_shop_pivot = pd.pivot_table(shopping_data, index=['User_id'], values=['Action'], aggfunc=len)
    on_shop_pivot = DataFrame(on_shop_pivot)
    on_shop_pivot.columns = ['on_shop']
    dataset = pd.merge(dataset, on_shop_pivot, on='User_id', how='left', left_index=True, sort=False)
    # 线上使用率
    ratio17 = on_shop_pivot['on_shop'].div(on_get_pivot['on_get'], axis=0)
    ratio17 = DataFrame(ratio17)
    ratio17.columns = ['ratio17']
    dataset = pd.merge(dataset, ratio17, on='User_id', how='left', left_index=True, sort=False)
    # 线上使用次数
    on_use_pivot = pd.pivot_table(use_cou_data, index=['User_id'], values=['Action'], aggfunc=len)
    on_use_pivot = pd.DataFrame(on_use_pivot)
    try:
        on_use_pivot.columns = ['on_use']
        dataset = pd.merge(dataset, on_use_pivot, on='User_id', how='left', left_index=True, sort=False)
    except:
        dataset['on_use'] = 0
    # 线下不使用占总不使用的比值
    off_nuse_data = offline_data[(offline_data.use == 0)].copy()
    on_nuse_data = online_data[(online_data.Coupon_id != 0) & (online_data.Date == 0)].copy()
    nuse1 = pd.pivot_table(off_nuse_data, index=['User_id'], values=['use'], aggfunc=len)
    nuse2 = pd.pivot_table(on_nuse_data, index=['User_id'], values=['Action'], aggfunc=len)
    nuse = pd.merge(nuse2, nuse1, on='User_id', how='outer', left_index=True, sort=False)
    nuse.fillna(0, inplace=True)
    nuse['nuse'] = nuse['use'] + nuse['Action']
    nuse.drop(['use', 'Action'], inplace=True, axis=1)
    dataset = pd.merge(dataset, nuse, on='User_id', how='left', left_index=True, sort=False)
    # 线下使用占总使用
    off_use_data = offline_data[(offline_data.use == 1)].copy()
    on_use_data = online_data[(online_data.Coupon_id != 0) & (online_data.Date != 0)].copy()
    use1 = pd.pivot_table(off_use_data, index=['User_id'], values=['use'], aggfunc=len)
    use2 = pd.pivot_table(on_use_data, index=['User_id'], values=['Action'], aggfunc=len)
    use = pd.merge(use2, use1, on='User_id', how='outer', left_index=True, sort=False)
    use.fillna(0, inplace=True)
    use['use'] = use['use'] + use['Action']
    use.drop(['use', 'Action'], inplace=True, axis=1)
    dataset = pd.merge(dataset, use, on='User_id', how='left', left_index=True, sort=False)
    # 线下领取占总领取
    on_get = online_data[(online_data.Coupon_id != 0)].copy()
    off_get = offline_data[(offline_data.Coupon_id != 0)].copy()
    get1 = pd.pivot_table(off_get, index=['User_id'], values=['use'], aggfunc=len)
    get2 = pd.pivot_table(on_get, index=['User_id'], values=['Action'], aggfunc=len)
    get = pd.merge(get1, get2, on='User_id', how='outer', left_index=True, sort=False)
    get.fillna(0, inplace=True)
    get['get'] = get['use'] + get['Action']
    get.drop(['use', 'Action'], inplace=True, axis=1)
    dataset = pd.merge(dataset, get, on='User_id', how='left', left_index=True, sort=False)
    dataset.index = data_index
    return dataset


def other_feature(dataset):
    dataset['front_get'] = 0
    dataset['back_get'] = 0
    dataset['front_same'] = 0
    dataset['back_same'] = 0
    dataset['front_time'] = 0
    dataset['back_time'] = 0
    dataset['same_cou'] = 0
    dataset['same_day'] = 0
    dataset['front_day_get_same'] = 0
    dataset['back_day_get_same'] = 0
    dataset['avg_get_day'] = 0
    dataset['first'] = 0
    dataset['last'] = 0
    dataset = dataset.sort_values(by=['User_id', 'Date_received'], axis=0, ignore_index=False)
    temp_data = dataset[(dataset.Coupon_id == 0)].copy()
    index = dataset[(dataset.Coupon_id == 0)].index.tolist()
    dataset.drop(index, inplace=True)
    front_time = 0
    front_user = -111346
    front_cou = 0
    count = 0
    mess_cou_list = []
    avg_day = 0
    get_front_user = -1
    get_front_ind = -1
    for i, mess in dataset.iterrows():
        if mess['User_id'] != get_front_user:  # 新用户第一次出现first置1
            dataset.loc[i, 'first'] = 1
            try:
                dataset.loc[get_front_ind, 'last'] = 1  # 上一个用户last置1
            except:
                pass
        get_front_user = mess['User_id']
        get_front_ind = i
        if mess['User_id'] == front_user:
            dataset.loc[i, 'front_get'] = count
            dataset.loc[i, 'back_get'] = data_sum - count - 1
            time_diff = mess['Date_received'] - front_time
            time_diff = time_diff.days
            dataset.loc[i, 'front_time'] = time_diff
            dataset.loc[front_ind, 'back_time'] = time_diff
            avg_day += time_diff
            count += 1
            if front_cou == mess['Coupon_id'] and front_time == mess['Date_received']:
                dataset.loc[i, 'same_day'] = 1
                dataset.loc[front_ind, 'same_day'] = 1
                dataset.loc[i, 'same_cou'] = 1
            elif front_cou == mess['Coupon_id']:
                dataset.loc[i, 'same_cou'] = 1
                dataset.loc[front_ind, 'same_cou'] = 1
            else:
                dataset.loc[i, 'same_cou'] = 0
                front_cou = mess['Coupon_id']
            mess_cou_list.append((i, mess['Coupon_id'], mess['Date_received']))
            front_ind = i
            front_time = mess['Date_received']
        else:
            t = dataset[(dataset.User_id == mess['User_id'])]
            data_sum = len(t)
            count = 1
            front_user = mess['User_id']
            front_time = mess['Date_received']
            front_cou = mess['Coupon_id']
            dataset.loc[i, 'front_time'] = 0
            dataset.loc[i, 'same_cou'] = 0
            dataset.loc[i, 'back_get'] = data_sum - 1
            try:
                dataset.loc[front_ind, 'back_time'] = 0
            except:
                pass
            front_ind = i
            length = len(mess_cou_list)
            try:
                avg_day = avg_day / length
            except:
                continue
            for j in range(length):
                ind = mess_cou_list[j][0]
                cou = mess_cou_list[j][1]
                day_time = mess_cou_list[j][2]
                dataset.loc[ind, 'avg_get_day'] = avg_day
                t_count = 0
                sign = True
                for k in range(j + 1, length):
                    if cou == mess_cou_list[k][1]:
                        t_count += 1
                        if sign:
                            diff = mess_cou_list[k][2] - day_time
                            diff = diff.days
                            dataset.loc[ind, 'back_day_get_same'] = diff
                            sign = False
                dataset.loc[ind, 'back_same'] = t_count
            for j in range(length - 1, -1, -1):
                ind = mess_cou_list[j][0]
                cou = mess_cou_list[j][1]
                day_time = mess_cou_list[j][2]
                t_count = 0
                sign = True
                for k in range(j - 1, -1, -1):
                    if cou == mess_cou_list[k][1]:
                        t_count += 1
                        if sign:
                            diff = mess_cou_list[k][2] - day_time
                            diff = diff.days
                            dataset.loc[ind, 'front_day_get_same'] = diff
                            sign = False
                dataset.loc[ind, 'front_same'] = t_count
            mess_cou_list = [(i, mess['Coupon_id'], mess['Date_received'])]
    dataset = dataset.append(temp_data)
    return dataset


def get_label(x):
    k = (x['Date'] - x['Date_received']).days
    if k <= 15:
        return 1
    else:
        return 0


def creat_cross_data():
    global test_data
    # feature_data 提取特征传给label_data
    feature_off_data1 = off_data[off_data['Date_received'].isin(pd.date_range('2016/01/01', periods=90)) |
                                 off_data['Date'].isin(pd.date_range('2016/01/01', periods=105))]
    feature_on_data1 = on_data[on_data['Date_received'].isin(pd.date_range('2016/01/01', periods=90)) |
                               on_data['Date'].isin(pd.date_range('2016/01/01', periods=105))]
    feature_off_data2 = off_data[off_data['Date_received'].isin(pd.date_range('2016/02/01', periods=90)) |
                                 off_data['Date'].isin(pd.date_range('2016/02/01', periods=105))]
    feature_on_data2 = on_data[on_data['Date_received'].isin(pd.date_range('2016/02/01', periods=90)) |
                               on_data['Date'].isin(pd.date_range('2016/02/01', periods=105))]
    feature_off_test = off_data[off_data['Date_received'].isin(pd.date_range('2016/03/16', periods=105)) |
                                off_data['Date'].isin(pd.date_range('2016/03/16', periods=105))]
    feature_on_test = on_data[on_data['Date_received'].isin(pd.date_range('2016/03/16', periods=105)) |
                              on_data['Date'].isin(pd.date_range('2016/03/16', periods=105))]
    # predict_label_data
    label_off_data1 = off_data[off_data['Date_received'].isin(pd.date_range('2016/04/14', periods=30))]
    label_off_data2 = off_data[off_data['Date_received'].isin(pd.date_range('2016/05/15', periods=30))]
    # 打标签
    label_off_data1['label'] = label_off_data1[['Date_received', 'Date']].apply(lambda x: get_label(x), axis=1)
    label_off_data2['label'] = label_off_data2[['Date_received', 'Date']].apply(lambda x: get_label(x), axis=1)
    feature_off_data1['label'] = feature_off_data1[['Date_received', 'Date']].apply(lambda x: get_label(x), axis=1)
    feature_off_data2['label'] = feature_off_data2[['Date_received', 'Date']].apply(lambda x: get_label(x), axis=1)
    feature_off_test['label'] = feature_off_test[['Date_received', 'Date']].apply(lambda x: get_label(x), axis=1)
    # 特征提取
    print('extract_feature')
    indexs = label_off_data1.index.tolist()
    label_off_data1 = get_feature(label_off_data1, feature_off_data1)
    label_off_data1.index = indexs
    label_off_data1 = get_on_feature(label_off_data1, feature_on_data1, feature_off_data1)
    label_off_data1 = other_feature(label_off_data1)
    print('train1_finish')
    indexs = label_off_data2.index.tolist()
    label_off_data2 = get_feature(label_off_data2, feature_off_data2)
    label_off_data2.index = indexs
    label_off_data2 = get_on_feature(label_off_data2, feature_on_data2, feature_off_data2)
    label_off_data2 = other_feature(label_off_data2)
    print('train_2_finish')
    indexs = test_data.index.tolist()
    test_data = get_feature(test_data, feature_off_test)
    test_data.index = indexs
    test_data = get_on_feature(test_data, feature_on_test, feature_off_test)
    test_data.index = indexs
    test_data = other_feature(test_data)
    print('test_finish')
    label_data = pd.concat([label_off_data1, label_off_data2], axis=0)
    return label_data


def judge_cal(x):
    week = str(x).replace(' 00:00:00', '').split('-')
    try:
        if is_workday(date(int(week[0]), int(week[1]), int(week[2]))):
            return 1
        else:
            return 0
    except:
        return 0


def judge_week(x):
    week = str(x).replace(' 00:00:00', '').split('-')
    try:
        day = date(int(week[0]), int(week[1]), int(week[2])).weekday() + 1
    except:
        day = 3
    return day


def which_day(x):
    week = str(x).replace(' 00:00:00', '').split('-')
    try:
        return int(week[2])
    except:
        return 0


def pretreatment():
    global off_data, test_data, on_data
    # 处理线下
    off_data['Distance'] = off_data['Distance'].fillna(11)
    # 修改时间格式
    off_data['Date_received'] = pd.to_datetime(off_data['Date_received'], format='%Y%m%d')
    off_data['Date'] = pd.to_datetime(off_data['Date'], format='%Y%m%d')
    # 判断周几
    off_data['week'] = off_data['Date_received']
    off_data['week'] = off_data['week'].apply(lambda x: judge_week(x))
    # 判断是一月的第几天
    off_data['month_day'] = off_data['Date_received'].apply(lambda x: which_day(x))
    # 判断是否休息
    off_data['work_day'] = off_data['Date_received']
    off_data['work_day'] = off_data['work_day'].apply(lambda x: judge_cal(x))

    # 处理在线
    on_data['Action'] = on_data['Action'].map(int)
    # 修改时间格式
    on_data['Date_received'] = pd.to_datetime(on_data['Date_received'], format='%Y%m%d')
    on_data['Date'] = pd.to_datetime(on_data['Date'], format='%Y%m%d')
    # 判断周几
    on_data['week'] = on_data['Date_received']
    on_data['week'] = on_data['week'].apply(lambda x: judge_week(x))
    # 判断是否休息
    on_data['work_day'] = on_data['Date_received']
    on_data['work_day'] = on_data['work_day'].apply(lambda x: judge_cal(x))

    # 处理测试集
    test_data['Distance'] = test_data['Distance'].fillna(11)
    # 修改时间格式
    test_data['Date_received'] = pd.to_datetime(test_data['Date_received'], format='%Y%m%d')
    # 判断周几
    test_data['week'] = test_data['Date_received']
    test_data['week'] = test_data['week'].apply(lambda x: judge_week(x))
    # 判断是一月的第几天
    test_data['month_day'] = test_data['Date_received'].apply(lambda x: which_day(x))
    # 判断是否休息
    test_data['work_day'] = test_data['Date_received']
    test_data['work_day'] = test_data['work_day'].apply(lambda x: judge_cal(x))


def model_xgb(train, test):
    # xgb参数
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'tree_method': 'gpu_hist',
              'min_child_weight': 0.93,  # 1.1
              'max_depth': 2,  # 3
              'lambda': 11,  # 11
              'gamma': 0.22,
              'subsample': 0.7,
              'colsample_bytree': 0.755,
              'colsample_bylevel': 0.71,
              'eta': 0.135,
              'nthread': 5,
              'predictor': 'gpu_predictor',
              'verbosity': 1
              }
    print(params)
    x_data = train.drop(['label'], axis=1).copy()
    label_data = train['label'].copy()
    best_feature = KBest_select(x_data, label_data, 120)
    x_data = train[best_feature]
    test = test[best_feature]
    train_data = xgb.DMatrix(x_data, label=label_data)
    test_data = xgb.DMatrix(test)
    watchlist = [(train_data, 'train')]
    model = xgb.train(params, train_data, 1300, watchlist)  # 1500
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
    model = SelectKBest(chi2, k=feature_count)
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
    data = pd.read_csv('data/tc/ccf_offline_stage1_train.csv')
    off_data = data
    data = pd.read_csv('data/tc/ccf_online_stage1_train.csv')
    on_data = data
    test_data = pd.read_csv('data/tc/ccf_offline_stage1_test_revised.csv')
    test_front = test_data[['User_id', 'Coupon_id', 'Date_received']].copy()
    pretreatment()
    # 划分数据集并提取特征
    print('划分并提取特征数据')
    Train_data = creat_cross_data()
    Train_data.dropna(subset=['User_id'], inplace=True)
    test_data.dropna(subset=['User_id'], inplace=True)
    # 存储数据
    print('存储数据')
    # Train_data.sort_index(inplace=True)
    test_data.sort_index(inplace=True)
    test_data['Date_received'] = test_front['Date_received']
    # 调整数据类型
    Train_data['Distance'] = Train_data['Distance'].map(int)
    test_data['Distance'] = test_data['Distance'].map(int)
    Train_data['User_id'] = Train_data['User_id'].map(int)
    test_data['User_id'] = test_data['User_id'].map(int)
    # 填充空值
    Train_data.fillna(0, inplace=True)
    test_data.fillna(0, inplace=True)
    Train_data.to_csv('all_Train_data.csv', index=False)
    test_data.to_csv('all_Test_data.csv', index=False)
    # ------------------------------------------------读入数据---------------------------------------------------
    Train_data = pd.read_csv('all_Train_data.csv')
    test_data = pd.read_csv('all_Test_data.csv')
    print('删除数据')
    # 去掉无用数据
    t_train = Train_data.copy()
    test_front = test_data[['User_id', 'Coupon_id', 'Date_received']].copy()
    labels = Train_data['label']
    Train_data.drop(['User_id', 'Coupon_id', 'Date_received', 'label', 'Date', 'Discount_rate', 'Merchant_id'], inplace=True, axis=1)
    test_data.drop(['User_id', 'Coupon_id', 'Date_received', 'Discount_rate', 'Merchant_id'], inplace=True, axis=1)
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
    xgb_train2()
    contact()
    #  程序最后会生成三个文件 xgb_predict.csv, xgb_predict2.csv, res.csv。
    #  其中res.csv为最终结果