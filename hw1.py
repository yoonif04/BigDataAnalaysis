
import numpy as np
import pandas as pd
from random import shuffle, seed

# 데이터 확인
d = pd.read_csv('train_mnist.csv')
d.info()
d.isnull()
np.percentile(d, [0, 25, 50, 75, 100])

# 데이터 형태 그려보기
import matplotlib.pyplot as plt
from UtilForHw import load_mnist
d_mn, l_mn = load_mnist('train_mnist.csv')
dig = d_mn[50].reshape((28, 28))
fig, ax = plt.subplots()
cax = ax.imshow(dig, cmap='gray')
ax.set_title('digit %d' %l_mn[50])
cbar = fig.colorbar(cax, ticks=[0, 255/2, 255])
cbar.ax.set_yticklabels(['0', '%d'%int(255/2), '255'])

# mnist data load using util
from UtilForHw import load_mnist, getxfold, runCVshuffle
d_mn, l_mn = load_mnist('train_mnist.csv')

# Decision Tree - default 사용
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus

clf_dt = DecisionTreeClassifier()
acc_dt = runCVshuffle(clf_dt, d_mn, l_mn)
mean_acc_dt = np.mean(acc_dt)

prfs_dt = runCVshuffle(clf_dt, d_mn, l_mn, isAcc=False)
prfs_dt_np = np.array(prfs_dt)
m_prfs_dt = []
for j in range(4):
    x = 0
    for i in range(10):
        x += sum(prfs_dt_np[i,j])
    m_prfs_dt.append(x/100)

export_graphviz(clf_dt, out_file="dtdtdt.gv")
graph = pydotplus.graph_from_dot_file("dtdtdt.gv")
graph.write_png("dtdtdt.png")

# Decision Tree 함수 내부 변경
clf_dt2 = DecisionTreeClassifier(max_depth=3)
acc_dt2 = runCVshuffle(clf_dt2, d_mn, l_mn)
mean_acc_dt2 = np.mean(acc_dt2)

prfs_dt2 = runCVshuffle(clf_dt, d_mn, l_mn, isAcc=False)
prfs_dt_np2 = np.array(prfs_dt2)
m_prfs_dt2 = []
for j in range(4):
    x = 0
    for i in range(10):
        x += sum(prfs_dt_np2[i,j])
    m_prfs_dt2.append(x/100)

export_graphviz(clf_dt2, out_file="dt2.gv")
graph = pydotplus.graph_from_dot_file("dt2.gv")
graph.write_png("dt2.png")

# Naive Bayesian Classifier - default 사용
from sklearn.naive_bayes import GaussianNB

clf_nb = GaussianNB()
acc_nb = runCVshuffle(clf_nb, d_mn, l_mn)
mean_acc_nb = np.mean(acc_nb)

prfs_nb = runCVshuffle(clf_nb, d_mn, l_mn, isAcc=False)
prfs_nb_np = np.array(prfs_nb)
m_prfs_nb = []
for j in range(4):
    x = 0
    for i in range(10):
        x += sum(prfs_nb_np[i,j])
    m_prfs_nb.append(x/100)

# k-Nearest Neighbor Classifier - default 사용 시간 오래 소요
from sklearn.neighbors import KNeighborsClassifier

clf_knn = KNeighborsClassifier()
acc_knn = runCVshuffle(clf_knn, d_mn, l_mn)
mean_acc_knn = np.mean(acc_knn)

prfs_knn = runCVshuffle(clf_knn, d_mn, l_mn, isAcc=False)
prfs_knn_np = np.array(prfs_knn)
m_prfs_knn = []
for j in range(4):
    x = 0
    for i in range(10):
        x += sum(prfs_knn_np[i,j])
    m_prfs_knn.append(x/100)


# Logistic Regression -default
from sklearn.linear_model import LogisticRegression as lr

clf_lr = lr()
acc_lr = runCVshuffle(clf_lr, d_mn, l_mn)
mean_acc_lr = np.mean(acc_lr)

prfs_lr = runCVshuffle(clf_lr, d_mn, l_mn, isAcc=False)
prfs_lr_np = np.array(prfs_lr)
m_prfs_lr = []
for j in range(4):
    x = 0
    for i in range(10):
        x += sum(prfs_lr_np[i,j])
    m_prfs_lr.append(x/100)

# Logistic Regression -C 조젇
clf_lr2 = lr(C=2)
acc_lr2 = runCVshuffle(clf_lr2, d_mn, l_mn)
mean_acc_lr2 = np.mean(acc_lr2)

prfs_lr2 = runCVshuffle(clf_lr2, d_mn, l_mn, isAcc=False)
prfs_lr2_np = np.array(prfs_lr2)
m_prfs_lr2 = []
for j in range(4):
    x = 0
    for i in range(10):
        x += sum(prfs_lr2_np[i,j])
    m_prfs_lr2.append(x/100)

# Perceptron - max_iter 조정
from sklearn.linear_model import Perceptron

clf_pc = Perceptron(max_iter=500, n_jobs=3)
acc_pc = runCVshuffle(clf_pc, d_mn, l_mn)
mean_acc_pc = np.mean(acc_pc)

prfs_pc = runCVshuffle(clf_pc, d_mn, l_mn, isAcc=False)
prfs_pc_np = np.array(prfs_pc)
m_prfs_pc = []
for j in range(4):
    x = 0
    for i in range(10):
        x += sum(prfs_pc_np[i,j])
    m_prfs_pc.append(x/100)

# Perceptron - max_iter, eta0 조정
clf_pc2 = Perceptron(max_iter=500, n_jobs=3, eta0=0.1)
acc_pc2 = runCVshuffle(clf_pc2, d_mn, l_mn)
mean_acc_pc2 = np.mean(acc_pc2)

prfs_pc2 = runCVshuffle(clf_pc2, d_mn, l_mn, isAcc=False)
prfs_pc2_np = np.array(prfs_pc2)
m_prfs_pc2 = []
for j in range(4):
    x = 0
    for i in range(10):
        x += sum(prfs_pc2_np[i,j])
    m_prfs_pc2.append(x/100)

# Multi-layer Perceptron - hidden_layer_sizes, max_iter 조정 시간소요 많음
from sklearn.neural_network import MLPClassifier as mlp

clf_mlp = mlp(hidden_layer_sizes=20, max_iter=500)
acc_mlp = runCVshuffle(clf_mlp, d_mn, l_mn)
mean_acc_mlp = np.mean(acc_mlp)

prfs_mlp = runCVshuffle(clf_mlp, d_mn, l_mn, isAcc=False)
prfs_mlp_np = np.array(prfs_mlp)
m_prfs_mlp = []
for j in range(4):
    x = 0
    for i in range(10):
        x += sum(prfs_mlp_np[i,j])
    m_prfs_mlp.append(x/100)

# Multi-layer Perceptron - hidden_layer_sizes, activation, learning_rate_init, max_iter 조정 시간소요 많음
clf_mlp2 = mlp(hidden_layer_sizes=20, max_iter=500, learning_rate_init=0.1, activation='tanh')
acc_mlp2 = runCVshuffle(clf_mlp2, d_mn, l_mn)
mean_acc_mlp2 = np.mean(acc_mlp2)

prfs_mlp2 = runCVshuffle(clf_mlp2, d_mn, l_mn, isAcc=False)
prfs_mlp2_np = np.array(prfs_mlp2)
m_prfs_mlp2 = []
for j in range(4):
    x = 0
    for i in range(10):
        x += sum(prfs_mlp2_np[i,j])
    m_prfs_mlp2.append(x/100)

# Random Forest
from sklearn.ensemble import RandomForestClassifier as rf
clf_rf = rf()
acc_rf = runCVshuffle(clf_rf, d_mn, l_mn)
mean_acc_rf = np.mean(acc_rf)

prfs_rf = runCVshuffle(clf_rf, d_mn, l_mn, isAcc=False)
prfs_rf_np = np.array(prfs_rf)
m_prfs_rf = []
for j in range(4):
    x = 0
    for i in range(10):
        x += sum(prfs_rf_np[i,j])
    m_prfs_rf.append(x/100)

# Linear Discriminant Analysis - n_component
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
clf_lda = lda(n_components=2)
acc_lda = runCVshuffle(clf_lda, d_mn, l_mn)
mean_acc_lda = np.mean(acc_lda)

prfs_lda = runCVshuffle(clf_lda, d_mn, l_mn, isAcc=False)
prfs_lda_np = np.array(prfs_lda)
m_prfs_lda = []
for j in range(4):
    x = 0
    for i in range(10):
        x += sum(prfs_lda_np[i,j])
    m_prfs_lda.append(x/100)

# Linear Discriminant Analysis - n_component 값 수정 (8까지 가능)
clf_lda2 = lda(n_components=8)
acc_lda2 = runCVshuffle(clf_lda2, d_mn, l_mn)
mean_acc_lda2 = np.mean(acc_lda2)

prfs_lda2 = runCVshuffle(clf_lda2, d_mn, l_mn, isAcc=False)
prfs_lda2_np = np.array(prfs_lda2)
m_prfs_lda2 = []
for j in range(4):
    x = 0
    for i in range(10):
        x += sum(prfs_lda2_np[i,j])
    m_prfs_lda2.append(x/100)

# Quadratic Discriminant Analysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
clf_qda = QDA()
acc_qda = runCVshuffle(clf_qda, d_mn, l_mn)
mean_acc_qda = np.mean(acc_qda)

prfs_qda = runCVshuffle(clf_qda, d_mn, l_mn, isAcc=False)
prfs_qda_np = np.array(prfs_qda)
m_prfs_qda = []
for j in range(4):
    x = 0
    for i in range(10):
        x += sum(prfs_qda_np[i,j])
    m_prfs_qda.append(x/100)

# Support Vector Machine - kernel, C, gamma, coef0 조정
from sklearn.svm import LinearSVC
clf_lsvc = LinearSVC()
acc_lsvc = runCVshuffle(clf_lsvc, d_mn, l_mn)
mean_acc_lsvc = np.mean(acc_lsvc)

prfs_lsvc = runCVshuffle(clf_lsvc, d_mn, l_mn, isAcc=False)
prfs_lsvc_np = np.array(prfs_lsvc)
m_prfs_lsvc = []
for j in range(4):
    x = 0
    for i in range(10):
        x += sum(prfs_lsvc_np[i,j])
    m_prfs_lsvc.append(x/100)

# GridSearch CV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
param1 = {'kernel': ['linear'], 'C': np.array([pow(10, x - 3) for x in range(6)])}
param2 = {'kernel': ['rbf'], 'C': np.array([pow(10, x - 3) for x in range(6)]),
          'gamma': np.array([pow(10, x - 3) for x in range(6)])}
param3 = {'kernel': ['poly', 'sigmoid'], 'C': np.array([pow(10, x - 3) for x in range(6)]),
          'gamma': np.array([pow(10, x - 3) for x in range(6)]),
          'coef0': np.array([pow(10, x - 3) for x in range(6)])}

svc = SVC()
seed(0)
numbers = list(range(len(d_mn)))
shuffle(numbers)
sd_mn = d_mn[numbers]
sl_mn = l_mn[numbers]

# parameter1 : linear
clf_p1 = GridSearchCV(svc, param1, cv=5, n_jobs=4)
clf_p1.fit(sd_mn, sl_mn)

prms_p1 = clf_p1.cv_results_['params']
acc_means_p1 = clf_p1.cv_results_['mean_test_score']
for mean, prm in zip(acc_means_p1, prms_p1):
    print("%0.3f for %r" % (mean, prm))


# parameter2 : rbf
clf_p2 = GridSearchCV(svc, param2, cv=5, n_jobs=4)
clf_p2.fit(sd_mn, sl_mn)

prms_p2 = clf_p2.cv_results_['params']
acc_means_p2 = clf_p2.cv_results_['mean_test_score']
for mean, prm in zip(acc_means_p2, prms_p2):
    print("%0.3f for %r" % (mean, prm))


# parameter3: poly, sigmoid
clf_p3 = GridSearchCV(svc, param3, cv=5, n_jobs=4)
clf_p3.fit(sd_mn, sl_mn)

prms_p3 = clf_p3.cv_results_['params']
acc_means_p3 = clf_p3.cv_results_['mean_test_score']
for mean, prm in zip(acc_means_p3, prms_p3):
    print("%0.3f for %r" % (mean, prm))


# PCA with sklearn
from sklearn.decomposition import PCA as skPCA
skpc = skPCA(n_components=8)
skpc.fit(d_mn)
tr3 = skpc.transform(d_mn)
skpc.explained_variance_ratio_

# Map to New Space using PC
import matplotlib.pyplot as plt

plt.tick_params(reset=True)
axis = ['PC1', 'PC2']
df_mn_pca = pd.DataFrame(tr3[:, :2], columns=axis)

df_mn_pca['target'] = l_mn
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
markers = ['o', 'v', '^', '<', '>', '*', 'D', 'h', 'x', 's']

for i, marker in enumerate(markers):
    x_axis_data = df_mn_pca[df_mn_pca['target'] == i]['PC1']
    y_axis_data = df_mn_pca[df_mn_pca['target'] == i]['PC2']
    plt.scatter(x_axis_data, y_axis_data, marker=marker, label=labels[i])
plt.legend(loc='upper right')
plt.xlabel("PC1")
plt.ylabel('PC2')
