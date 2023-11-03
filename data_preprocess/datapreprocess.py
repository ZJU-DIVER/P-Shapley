import pickle

import numpy
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

trainsize = 500
devsize = trainsize
testsize = trainsize * 4
dataset = "click"
orgdata = arff.loadarff('../data/{}.arff'.format(dataset))
print(orgdata)
data, _ = orgdata
idx = np.arange(len(data))
sliceidx = numpy.random.choice(idx, trainsize+devsize+testsize, replace=False)
data = data[sliceidx]

arr = np.array(data)
data = np.array(arr.tolist())
dataX = np.delete(data, -1, 1)
le = LabelEncoder()
# 对dataX中的每一列进行标签编码
for i in range(dataX.shape[1]):
    if not np.issubdtype(dataX[:, i].dtype, np.number):
        dataX[:, i] = le.fit_transform(dataX[:, i])
dataX = np.array(dataX, dtype=float)
print(dataX.shape)
dataY = data[:, -1]
# 多分类到二分类
dataY = le.fit_transform(dataY)
dataY = [1 if i == 1 else 0 for i in dataY]
print(np.sum(dataY), dataY[:10])

# # 采样
# tmpX, tstX, tmpY, tstY = train_test_split(dataX, dataY, test_size = 0.6666666, stratify = 0.5)
# trnX, devX, trnY, devY = train_test_split(tmpX, tmpY, test_size = 0.5, stratify = 0.5)
#
# print(np.sum(trnY), trnY[:10])
tmpX, tstX, tmpY, tstY = train_test_split(dataX, dataY, test_size = 0.6666666, stratify = dataY)
trnX, devX, trnY, devY = train_test_split(tmpX, tmpY, test_size = 0.5, stratify = tmpY)

result = {}
result["trnX"] = trnX
result["trnY"] = trnY
result["devX"] = devX
result["devY"] = devY
result["tstX"] = tstX
result["tstY"] = tstY
with open("../data/{}_{}.pkl".format(dataset, trainsize), "wb") as f:
    pickle.dump(result, f)

# Train a logistic regression classifier and evaluate its accuracyc
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
clf = LR(solver="liblinear", max_iter=500, random_state=0)
clf.fit(trnX, trnY)
acc = accuracy_score(clf.predict(tstX), tstY)
print(acc)