
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from random import shuffle, seed
from UtilForMid import runCVshuffle, featurenum, runCV

usecol = [x for x in range(2, 40)]
telco = pd.read_csv('telco.csv', usecols=usecol, na_values='?').dropna()

# object & age to binary features
# 성별
uniq_gender = set(telco['성별'])
only_women = telco['성별'] == '여'
only_men = telco['성별'] == '남'

np_ow = only_women.astype(np.int32).values
np_om = only_men.astype(np.int32).values

np_gender = np.vstack((np_ow, np_om))
np_gender = np.transpose(np_gender)

# 연령
only_10 = (10 <= telco['연령']) == (telco['연령'] < 20)
only_20 = (20 <= telco['연령']) == (telco['연령'] < 30)
only_30 = (30 <= telco['연령']) == (telco['연령'] < 40)
only_40 = (40 <= telco['연령']) == (telco['연령'] < 50)
only_50 = (50 <= telco['연령']) == (telco['연령'] < 60)
only_60 = (60 <= telco['연령']) == (telco['연령'] < 70)
only_70 = (70 <= telco['연령']) == (telco['연령'] < 80)
only_80 = 80 <= telco['연령']

np_o10 = only_10.astype(np.int32).values
np_o20 = only_20.astype(np.int32).values
np_o30 = only_30.astype(np.int32).values
np_o40 = only_40.astype(np.int32).values
np_o50 = only_50.astype(np.int32).values
np_o60 = only_60.astype(np.int32).values
np_o70 = only_70.astype(np.int32).values
np_o80 = only_80.astype(np.int32).values

np_age = np.vstack((np_o10, np_o20, np_o30, np_o40, np_o50, np_o60, np_o70, np_o80))
np_age = np.transpose(np_age)

# 요금제
uniq_callingplan = list(set(telco['요금제']))
l_np_cplan = []
for each_cplan in uniq_callingplan:
    only_cplan = telco['요금제'] == each_cplan

    np_cplan = only_cplan.astype(np.int32).values
    l_np_cplan.append(np_cplan)

np_cplan = np.vstack(l_np_cplan)
np_cplan = np.transpose(np_cplan)

# 핸드셋
uniq_hand = list(set(telco['핸드셋']))
l_np_hand = []
for each_handset in uniq_hand:
    only_hand = telco['핸드셋'] == each_handset

    np_hand = only_hand.astype(np.int32).values
    l_np_hand.append(np_hand)

np_handset = np.vstack(l_np_hand)
np_handset = np.transpose(np_handset)

# 통화량 구분
uniq_callvolume = list(set(telco['통화량구분']))
l_np_callvol = []
for each_callvol in uniq_callvolume:
    only_callvol = telco['통화량구분'] == each_callvol

    np_callvol = only_callvol.astype(np.int32).values
    l_np_callvol.append(np_callvol)

np_callvol = np.vstack(l_np_callvol)
np_callvol = np.transpose(np_callvol)

# 납부여부
uniq_pay = list(set(telco['납부여부']))
l_np_pay = []
for each_pay in uniq_pay:
    only_pay = telco['납부여부'] == each_pay

    np_pay = only_pay.astype(np.int32).values
    l_np_pay.append(np_pay)

np_pay = np.vstack(l_np_pay)
np_pay = np.transpose(np_pay)

# 통화품질불만
uniq_quality = set(telco['통화품질불만'])
only_qt = telco['통화품질불만'] == 'T'
only_qf = telco['통화품질불만'] == 'F'

np_oqt = only_qt.astype(np.int32).values
np_oqf = only_qf.astype(np.int32).values

np_quality = np.vstack((np_oqt, np_oqf))
np_quality = np.transpose(np_quality)

# binary features
binary_features = np.column_stack([np_gender, np_age, np_cplan, np_handset, np_callvol, np_pay, np_quality])

# 이탈여부 -> target
uniq_leave = set(telco['이탈여부'])
only_lt = telco['이탈여부'] == '이탈'
only_lf = telco['이탈여부'] == '유지'

np_olt = only_lt.astype(np.int32).values
np_olf = only_lf.astype(np.int32).values

np_leave = np.vstack((np_olt, np_olf))
np_leave = np.transpose(np_leave)
np_leave_onecol = np.transpose(np_olt)

# 이산형
# 단선횟수
np_disconnection = telco['단선횟수'].values
# 주간통화횟수
np_callnum_week = telco['주간통화횟수'].values
# 야간통화횟수
np_callnum_night = telco['야간통화횟수'].values
# 주말통화횟수
np_callnum_weekend = telco['주말통화횟수'].values
# 국내통화횟수
np_callnum_domestic = telco['국내통화횟수'].values

discrete_telco = np.column_stack([np_disconnection, np_callnum_week, np_callnum_night, np_callnum_weekend, np_callnum_domestic])

# binary features 합치기
all_binary_features = np.column_stack([binary_features, discrete_telco])


# 연속형
# 서비스기간, 주간통화시간_분, 야간통화시간_분, 주말통화시간_분, 국제통화시간_분
# 국내통화요금_분, 국내통화시간_분, 요금부과시간, 분당통화요금, 총통화요금, 부과요금, 국제통화비율

# 연속형 변수들의 pairplot
usecol = [5,  12,  14,  16, 17, 18, 23, 27, 28, 30, 31, 37]
con_telco = pd.read_csv('telco.csv', usecols=usecol, na_values='?').dropna()
con_telco.corr()

con_telco.columns = ['svduration', 'calltimeweek', 'calltimenight', 'calltimeweekend', 'calltimeex', 'incost',
                     'calltimein', 'costtime', 'costpermin', 'costtotal', 'charge', 'exratio']
d = [5, 6, 7, 9, 10]
con_telco = con_telco.drop(con_telco.columns[d], axis=1)
sb.pairplot(con_telco, kind='reg')
plt.figure(figsize=(8, 8))
sb.heatmap(data=con_telco.corr(), annot=True, fmt='.2f', linewidths=.5, cmap='Reds')


# Normalize Continuous Features
import matplotlib.pyplot as plt

# 서비스기간, 0,1 normalization
plt.hist(telco['서비스기간'].values)
np_svduration = np.log(telco['서비스기간'] + 10)
np_svduration = (np_svduration-min(np_svduration))/(max(np_svduration)-min(np_svduration))
plt.hist(np_svduration)
plt.xlabel('sv_duration')

# 주간통화시간_분, log 씌워서 보정 (but, log0 음의 무한대 ->10을 더해줌)+ 0,1 normalization
plt.hist(telco['주간통화시간_분'].values)
np_calltimeweek = np.log(telco['주간통화시간_분'].values + 10)
np_calltimeweek = (np_calltimeweek- min(np_calltimeweek))/(max(np_calltimeweek)-min(np_calltimeweek))
plt.hist(np_calltimeweek)
plt.xlabel('calltime_week')

# 야간통화시간_분, log 씌워서 보정 (but, log0 음의 무한대 ->10을 더해줌)+ 0,1 normalization
plt.hist(telco['야간통화시간_분'].values)
np_calltimenight = np.log(telco['야간통화시간_분'].values + 10)
np_calltimenight = (np_calltimenight - min(np_calltimenight))/(max(np_calltimenight)-min(np_calltimenight))
plt.hist(np_calltimenight)
plt.xlabel('calltime_night')

# 주말통화시간_분, log 씌워서 보정 (but, log0 음의 무한대 ->10을 더해줌)+ 0,1 normalization
plt.hist(telco['주말통화시간_분'].values)
np_calltimeweekend = np.log(telco['주말통화시간_분'].values + 10)
np_calltimeweekend = (np_calltimeweekend - min(np_calltimeweekend))/(max(np_calltimeweekend)-min(np_calltimeweekend))
plt.hist(np_calltimeweekend)
plt.xlabel('calltime_weekend')

# 국제통화시간_분, log 씌워서 보정 (but, log0 음의 무한대 ->10을 더해줌)+ 0,1 normalization
plt.hist(telco['국제통화시간_분'].values)
np_calltimeex = np.log(telco['국제통화시간_분'].values + 10)
np_calltimeex = (np_calltimeex - min(np_calltimeex))/(max(np_calltimeex)-min(np_calltimeex))
plt.hist(np_calltimeex)
plt.xlabel('calltime_ex')

# # 국내통화요금_분, log 씌워서 보정 (but, log0 음의 무한대 ->1을 더해줌)+ 0,1 normalization
# plt.hist(telco['국내통화요금_분'].values)
# np_incost = np.log(telco['국내통화요금_분'].values + 1)
# np_incost = (np_incost-min(np_incost))/(max(np_incost)-min(np_incost))
# plt.hist(np_incost)
#
# # 국내통화시간_분, log 씌워서 보정 (but, log0 음의 무한대 ->10을 더해줌)+ 0,1 normalization
# plt.hist(telco['국내통화시간_분'].values)
# np_calltimein = np.log(telco['국내통화시간_분'].values + 10)
# np_calltimein = (np_calltimein - min(np_calltimein))/(max(np_calltimein)-min(np_calltimein))
# plt.hist(np_calltimein)

# # 요금부과시간 0,1 normalization
# plt.hist(telco['요금부과시간'].values)
# np_costtime = telco['요금부과시간'].values
# np_costtime = (np_costtime - min(np_costtime))/(max(np_costtime)-min(np_costtime))
# plt.hist(np_costtime)

# 분당통화요금 , log 씌워서 보정 (but, log0 음의 무한대 ->10을 더해줌)+ 0,1 normalization
plt.hist(telco['분당통화요금'].values)
np_costpermin = np.log(telco['분당통화요금'].values + 10)
np_costpermin = (np_costpermin - min(np_costpermin))/(max(np_costpermin)-min(np_costpermin))
plt.hist(np_costpermin)
plt.xlabel("costpermin")

# # 총통화요금 , log 씌워서 보정 (but, log0 음의 무한대 ->10을 더해줌)+ 0,1 normalization
# plt.hist(telco['총통화요금'].values)
# np_costtotal = np.log(telco['총통화요금'].values + 10)
# np_costtotal = (np_costtotal - min(np_costtotal))/(max(np_costtotal)-min(np_costtotal))
# plt.hist(np_costtotal)
#
# # 부과요금 , log 씌워서 보정 (but, log0 음의 무한대 ->10을 더해줌)+ 0,1 normalization
# plt.hist(telco['부과요금'].values)
# np_charge = np.log(telco['부과요금'].values + 10)
# np_charge = (np_charge - min(np_charge))/(max(np_charge)-min(np_charge))
# plt.hist(np_charge)

# 국제통화비율 - log 씌워서 보정 (but, log0 음의 무한대 ->10을 더해줌)+ 0,1 normalization
plt.hist(telco['국제통화비율'].values)
np_exratio = np.log(telco['국제통화비율'].values * 100)
np_exratio = (np_exratio - min(np_exratio))/(max(np_exratio) - min(np_exratio))
plt.hist(np_exratio)
plt.xlabel('ex_ratio')


# Merge continuous Feature
continuous_telco = np.column_stack([np_svduration, np_calltimeweek, np_calltimenight, np_calltimeweekend, np_calltimeex, np_costpermin, np_exratio])

df_continuous_telco = pd.DataFrame(continuous_telco)
sb.pairplot(df_continuous_telco)

# Merge all Feature
all_telco = np.column_stack([all_binary_features, continuous_telco])


# Analysis
# Decision Tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus

clf_dt = DecisionTreeClassifier()
acc_dt, c_index_dt = runCVshuffle(clf_dt, all_telco, np_leave)
mean_acc_dt = np.mean(acc_dt)

# k-Nearest Neighbor Classifier
from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier()
acc_knn, c_index_knn = runCVshuffle(clf_knn, all_telco, np_leave)
mean_acc_knn = np.mean(acc_knn)

# Logistic Regression
from sklearn.linear_model import LogisticRegression as lr
clf_lr = lr()
acc_lr, c_index_lr = runCV(clf_lr, all_telco, np_leave_onecol)
mean_acc_lr = np.mean(acc_lr)

# Perceptron
from sklearn.linear_model import Perceptron
clf_pc = Perceptron()
acc_pc, c_index_pc = runCV(clf_pc, all_telco, np_leave_onecol)
mean_acc_pc = np.mean(acc_pc)

# Random Forest
from sklearn.ensemble import RandomForestClassifier as rf
clf_rf = rf()
acc_rf, c_index_rf = runCVshuffle(clf_rf, all_telco, np_leave)
mean_acc_rf = np.mean(acc_rf)

# Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
clf_lda = lda()
acc_lda, c_index_lda = runCV(clf_lda, all_telco, np_leave_onecol)
mean_acc_lda = np.mean(acc_lda)

# Quadratic Discriminant Analysis  error: collinear
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
clf_qda = QDA()
acc_qda, c_index_qda = runCV(clf_qda, all_telco, np_leave_onecol)
mean_acc_qda = np.mean(acc_qda)

# Support Vector Machine - kernel, C, gamma, coef0 조정
from sklearn.svm import LinearSVC
clf_lsvc = LinearSVC()
acc_lsvc, c_index_svc = runCV(clf_lsvc, all_telco, np_leave_onecol)
mean_acc_lsvc = np.mean(acc_lsvc)

# Multi-layer Perceptron - hidden_layer_sizes, max_iter 조정
from sklearn.neural_network import MLPClassifier as mlp
clf_mlp = mlp()
acc_mlp, c_index_mlp = runCVshuffle(clf_mlp, all_telco, np_leave)
mean_acc_mlp = np.mean(acc_mlp)

feature_num_dt_mlp, c_data_mlp = featurenum(np_age, c_index_mlp, kind=8)

# GridSearch CV - mlp
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.model_selection import GridSearchCV
par1 = {'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['adam'],
        'hidden_layer_sizes': np.array([100, 200, 300, 400, 500]), 'learning_rate_init': np.array([0.001, 0.01, 0.1]),
        'power_t': np.array([0.3, 0.4, 0.5, 0.6]), 'max_iter': np.array([100, 200, 300, 400, 500])}

mlp = mlp()
seed(0)
numbers = list(range(len(all_telco)))
shuffle(numbers)
sd_telco = all_telco[numbers]
sl_telco = np_leave[numbers]

# parameter1
clf_p1 = GridSearchCV(mlp, par1, cv=5, n_jobs=4)
clf_p1.fit(sd_telco, sl_telco)

prms_p1 = clf_p1.cv_results_['params']
acc_means_p1 = clf_p1.cv_results_['mean_test_score']
for mean, prm in zip(acc_means_p1, prms_p1):
    if mean >= 0.909:
        print("%0.5f for %r" % (mean, prm))


# parameter2
par2 = {'activation': ['logistic', 'tanh'], 'solver': ['adam'], 'hidden_layer_sizes': np.array([50, 60, 70, 80, 90, 100]),
        'learning_rate_init': np.array([0.0005, 0.00075, 0.001]), 'power_t': np.array([0.1, 0.2, 0.3]),
        'max_iter': np.array([50, 60, 70, 80, 90, 100])}

mlp2 = mlp()
clf_p2 = GridSearchCV(mlp2, par2, cv=5, n_jobs=4)
clf_p2.fit(sd_telco, sl_telco)

prms_p2 = clf_p2.cv_results_['params']
acc_means_p2 = clf_p2.cv_results_['mean_test_score']
for mean, prm in zip(acc_means_p2, prms_p2):
    if mean >= 0.908:
        print("%0.5f for %r" % (mean, prm))


# GridSearch CV - rf
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.model_selection import GridSearchCV
par1_rf = {'criterion': ['gini', 'entropy'], 'n_estimators': [80, 90, 100, 110, 120],
        'max_features': ['log2', 'auto', 0.1, 0.2, 0.3], 'bootstrap': ['TRUE', 'FALSE']}

rf = rf()
seed(0)
numbers = list(range(len(all_telco)))
shuffle(numbers)
sd_telco = all_telco[numbers]
sl_telco = np_leave[numbers]

# parameter1
clf_p1_rf = GridSearchCV(rf, par1_rf, cv=5, n_jobs=4)
clf_p1_rf.fit(sd_telco, sl_telco)

prms_p1_rf = clf_p1_rf.cv_results_['params']
acc_means_p1_rf = clf_p1_rf.cv_results_['mean_test_score']
for mean, prm in zip(acc_means_p1_rf, prms_p1_rf):
    if mean >= 0.92:
        print("%0.5f for %r" % (mean, prm))

# parameter 2
par2_rf = {'criterion': ['gini', 'entropy'], 'n_estimators': [80, 90, 100, 110, 120],
        'max_features': [0.3, 0.4, 0.5, 0.6], 'bootstrap': ['TRUE', 'FALSE']}

rf2 = rf()
clf_p2_rf = GridSearchCV(rf2, par2_rf, cv=5, n_jobs=4)
clf_p2_rf.fit(sd_telco, sl_telco)

prms_p2_rf = clf_p2_rf.cv_results_['params']
acc_means_p2_rf = clf_p2_rf.cv_results_['mean_test_score']
for mean, prm in zip(acc_means_p2_rf, prms_p2_rf):
    if mean >= 0.9208:
        print("%0.5f for %r" % (mean, prm))

# parameter 3
par3_rf = {'criterion': ['gini', 'entropy'], 'n_estimators': [80, 90, 100, 110, 120],
        'max_features': [0.7, 0.8, 0.9], 'bootstrap': ['TRUE', 'FALSE']}

rf3 = rf()
clf_p3_rf = GridSearchCV(rf3, par3_rf, cv=5, n_jobs=4)
clf_p3_rf.fit(sd_telco, sl_telco)

prms_p3_rf = clf_p3_rf.cv_results_['params']
acc_means_p3_rf = clf_p3_rf.cv_results_['mean_test_score']
for mean, prm in zip(acc_means_p3_rf, prms_p3_rf):
    if mean >= 0.9208:
        print("%0.5f for %r" % (mean, prm))


# Random Forest -> Grid Search 결과 이용
from sklearn.ensemble import RandomForestClassifier as rf
clf_rf = rf(max_features=0.6, n_estimators=120, bootstrap='FALSE', criterion='gini')
acc_rf, c_index_rf = runCVshuffle(clf_rf, all_telco, np_leave)
mean_acc_rf = np.mean(acc_rf)

# 카테고리형의 특징
f_n_age, c_data_age = featurenum(np_age, c_index_rf, 8)
f_n_age = pd.DataFrame(f_n_age).transpose()
f_n_age.columns = ['10대', '20대', '30대', '40대', '50대', '60대', '70대', '80대']
age = np.sum(np_age, axis=0)
f_n_age/age

f_n_gen, c_data_gen = featurenum(np_gender, c_index_rf, 2)
f_n_gen = pd.DataFrame(f_n_gen).transpose()
f_n_gen.columns = ['여', '남']

f_n_cp, c_data_cp = featurenum(np_cplan, c_index_rf, 5)
f_n_cp = pd.DataFrame(f_n_cp).transpose()
f_n_cp.columns = ['Play 100', 'CAT 200', 'CAT 50', 'CAT 100', 'Play 300']
cp = sum(telco['요금제'] == 'CAT 200')
ratio_cp = f_n_cp/cp

f_n_hand, c_data_hand = featurenum(np_handset, c_index_rf, 11)
f_n_hand = pd.DataFrame(f_n_hand).transpose()
f_n_hand.columns = [uniq_hand]
hand = np.sum(np_handset, axis=0)
ratio_hand = f_n_hand/hand

f_n_cv, c_data_cv = featurenum(np_callvol, c_index_rf, 5)
f_n_cv = pd.DataFrame(f_n_cv).transpose()
f_n_cv.columns = [uniq_callvolume]
cv = np.sum(np_callvol, axis=0)
ratio_cv = f_n_cv/cv

f_n_pay, c_data_pay = featurenum(np_pay, c_index_rf, 4)
f_n_pay = pd.DataFrame(f_n_pay).transpose()
f_n_pay.columns = [uniq_pay]
pay = np.sum(np_pay, axis=0)
ratio_pay = f_n_pay/pay

f_n_quality, c_data_quality = featurenum(np_quality, c_index_rf, 2)
f_n_quality = pd.DataFrame(f_n_quality).transpose()
f_n_quality.columns = ['T', 'F']
quality = np.sum(np_quality, axis=0)
ratio_quality = f_n_quality/quality

# 이산형의 특징
c_data_disc = np_disconnection[c_index_rf]
f_n_disc = pd.value_counts(c_data_disc)
disc = pd.value_counts(np_disconnection)
ratio_disc = f_n_disc / disc
plt.scatter(f_n_disc.index, f_n_disc.values)

c_data_cnweek = np_callnum_week[c_index_rf]
f_n_cnweek = pd.value_counts(c_data_cnweek, sort=True)
plt.scatter(f_n_cnweek.index, f_n_cnweek.values)
plt.xlabel('call_num_week')
plt.ylabel('frequency')

c_data_cnnight = np_callnum_night[c_index_rf]
f_n_cnnight = pd.value_counts(c_data_cnnight, sort=True)
plt.scatter(f_n_cnnight.index, f_n_cnnight.values)
plt.xlabel('call_num_night')
plt.ylabel('frequency')

c_data_cnweekend = np_callnum_weekend[c_index_rf]
f_n_cnweekend = pd.value_counts(c_data_cnweekend, sort=True)
plt.scatter(f_n_cnweekend.index, f_n_cnweekend.values)
plt.xlabel('call_num_weekend')
plt.ylabel('frequency')

c_data_cndomestic = np_callnum_domestic[c_index_rf]
f_n_cndomestic = pd.value_counts(c_data_cndomestic, sort=True)
plt.scatter(f_n_cndomestic.index, f_n_cndomestic.values)
plt.xlabel('call_num_domestic')
plt.ylabel('frequency')


# 연속형의 특징
c_data_sv = np_svduration[c_index_rf]
plt.hist(c_data_sv)
c_data_ctweek = np_calltimeweek[c_index_rf]
plt.hist(c_data_ctweek)
c_data_ctnight = np_calltimenight[c_index_rf]
plt.hist(c_data_ctnight)
c_data_ctweekend = np_calltimeweekend[c_index_rf]
plt.hist(c_data_ctweekend)
c_data_ctex = np_calltimeex[c_index_rf]
plt.hist(c_data_ctex)
c_data_cpmin = np_costpermin[c_index_rf]
plt.hist(c_data_cpmin)
c_data_exratio = np_exratio[c_index_rf]
plt.hist(c_data_exratio)