from math import factorial
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, f1_score
from sklearn.preprocessing import LabelEncoder
import inspect, pickle
import numpy as np
import pandas as pd
from glob import glob
import _pickle as pkl
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from scipy.linalg import eigh
from scipy.stats import kendalltau, ttest_ind
from sklearn.cluster import KMeans

seed = 2024


def get_seed():
    return seed


def set_seed(x):
    global seed
    seed = x


def get_model(mode, **kwargs):
    '''
    Define a model to be used in computation of data values
    '''
    if inspect.isclass(mode):
        assert getattr(mode, 'fit', None) is not None, 'Custom model family should have a fit() method'
        model = mode(**kwargs)
    elif mode == 'LR':
        model = LogisticRegression(solver="liblinear", max_iter=5000, random_state=get_seed())
    # elif mode=='logistic':
    #     solver = kwargs.get('solver', 'liblinear')
    #     n_jobs = kwargs.get('n_jobs', -1)
    #     C = kwargs.get('C', 0.05) # 1.
    #     max_iter = kwargs.get('max_iter', 5000)
    #     model = LogisticRegression(solver=solver, n_jobs=n_jobs, C=C,
    #                                max_iter=max_iter, random_state=666)
    elif mode == 'linear':
        n_jobs = kwargs.get('n_jobs', -1)
        model = LinearRegression(n_jobs=n_jobs)
    elif mode == 'ridge':
        alpha = kwargs.get('alpha', 1.0)
        model = Ridge(alpha=alpha, random_state=666)
    elif mode == 'Tree':
        model = DecisionTreeClassifier(random_state=666)
    elif mode == 'RandomForest':
        n_estimators = kwargs.get('n_estimators', 50)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=666)
    elif mode == 'GB':
        n_estimators = kwargs.get('n_estimators', 50)
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=666)
    elif mode == 'AdaBoost':
        n_estimators = kwargs.get('n_estimators', 50)
        model = AdaBoostClassifier(n_estimators=n_estimators, random_state=666)
    elif mode == 'SVC':
        kernel = kwargs.get('kernel', 'rbf')
        C = kwargs.get('C', 0.5)  # 1.
        max_iter = kwargs.get('max_iter', 5000)
        model = SVC(probability=True, kernel=kernel, max_iter=max_iter, C=C, random_state=42)
    elif mode == 'SVC-GS':
        kernel = kwargs.get('kernel', 'rbf')
        C = kwargs.get('C', 0.05)  # 1.
        max_iter = kwargs.get('max_iter', 5000)
        model = SVC(probability=True, kernel=kernel, max_iter=max_iter, C=C, random_state=666)
    elif mode == 'LinearSVC':
        C = kwargs.get('C', 0.05)  # 1.
        max_iter = kwargs.get('max_iter', 5000)
        model = LinearSVC(loss='hinge', max_iter=max_iter, C=C, random_state=666)
    elif mode == 'GP':
        model = GaussianProcessClassifier(random_state=666)
    elif mode == 'KNN':
        n_neighbors = kwargs.get('n_neighbors', 5)
        n_jobs = kwargs.get('n_jobs', -1)
        model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs)
    elif mode == 'NB':
        model = MultinomialNB()
    else:
        raise ValueError("Invalid mode!")
    return model


def get_dataset(dataset, path, noise=False):
    if (dataset == "gaussian"):
        return load_classification_dataset()
    else:
        data = pickle.load(open(path, 'rb'))
        # The set of examples that will be evaluated
        trnX, trnY = data['trnX'], data['trnY']
        # We need a development set to value the training examples
        devX, devY = data['devX'], data['devY']
        tstX, tstY = data['tstX'], data['tstY']
        # 使用LabelEncoder将标签编码为0-1
        le = LabelEncoder()
        trnY = le.fit_transform(trnY)
        devY = le.transform(devY)
        tstY = le.transform(tstY)
        print("Loading dataset {} ...".format(dataset))
        print("Dimensions of data: {}".format(trnX.shape[1]))
        print("Number of training examples: {}".format(trnX.shape[0]))
        print("Labels: ", trnY[:10], devY[:10], tstY[:10])
        print("Labels : ", sum(trnY), len(trnY) - sum(trnY), sum(devY), len(devY) - sum(devY), sum(tstY),
              len(tstY) - sum(tstY))
        print("Number of validation/development examples: {}/{}".format(devX.shape[0], tstX.shape[0]))
        if noise:
            num = len(list(set(trnY)))
            for i in range(len(trnY)):
                if i % 10 == 1:
                    trnY[i] = (trnY[i] + 1) % num
        return trnX, trnY, devX, devY, tstX, tstY


def load_classification_dataset(n_data_to_be_valued=200,
                                n_val=100,
                                n_test=1000,
                                rid=1,
                                dataset='gaussian',
                                clf_path='clf_path'):
    '''
    This function loads classification (or density estimation) datasets for the point addition experiments.
    n_data_to_be_valued: The number of data points to be valued.
    n_val: the number of data points for evaluation of the utility function.
    n_test: the number of data points for evaluation of performances in point addition experiments.
    clf_path: path to classification datasets.
    '''
    if dataset == 'gaussian':
        print('-' * 50)
        print('GAUSSIAN-C')
        print('-' * 50)
        n, input_dim = 50000, 5
        data = np.random.normal(size=(n, input_dim))
        beta_true = np.array([5.0, 4.0, 3.0, 2.0, 1.0]).reshape(input_dim, 1)
        p_true = np.exp(data.dot(beta_true)) / (1. + np.exp(data.dot(beta_true)))
        target = np.random.binomial(n=1, p=p_true).reshape(-1)
    else:
        assert False, f"Check {dataset}"

    idxs = np.random.permutation(len(data))
    data, target = data[idxs], target[idxs]

    X = data[:n_data_to_be_valued]
    y = target[:n_data_to_be_valued]
    X_val = data[n_data_to_be_valued:(n_data_to_be_valued + n_val)]
    y_val = target[n_data_to_be_valued:(n_data_to_be_valued + n_val)]
    X_test = data[(n_data_to_be_valued + n_val):(n_data_to_be_valued + n_val + n_test)]
    y_test = target[(n_data_to_be_valued + n_val):(n_data_to_be_valued + n_val + n_test)]

    print(f'number of samples: {len(X)}')
    X_mean, X_std = np.mean(X, 0), np.std(X, 0)
    normalizer_fn = lambda x: (x - X_mean) / np.clip(X_std, 1e-12, None)
    X, X_val, X_test = normalizer_fn(X), normalizer_fn(X_val), normalizer_fn(X_test)

    return X, y, X_val, y_val, X_test, y_test


def m_cross_entropy(y_true, y_prob):
    # Only one class in y_true for each call. Lead to error
    # if len(y_prob[0]) <= 2:
    #     return log_loss(y_true, y_prob)
    # else:
    #     # multi-class classification
    return np.sum([np.log(y_prob[i, y_true[i]]) for i in range(len(y_true))])


def get_utility(x_train, y_train, x_valid, y_valid, clf, metric='accuracy'):
    """_summary_

    Args:
        x_train (_type_): _description_
        y_train (_type_): _description_
        x_valid (_type_): _description_
        y_valid (_type_): _description_
        clf (_type_): _description_

    Returns:
        _type_: _description_
    """

    if metric == 'accuracy':
        try:
            clf.fit(x_train, y_train)
            acc = accuracy_score(y_valid, clf.predict(x_valid))
        except ValueError:
            # Training set only has a single class
            acc = accuracy_score(y_valid, [y_train[0]] * len(y_valid))
    elif metric == 'auc':
        try:
            clf.fit(x_train, y_train)
            y_prob = clf.predict_proba(x_valid)[:, 1]
            acc = roc_auc_score(y_valid, y_prob)
        except ValueError:
            acc = 0.5
    elif metric == 'f1-score':
        try:
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_valid)
            acc = f1_score(y_valid, y_pred)
        except ValueError:
            # Handle potential errors, for example, when only one class is present
            rnd_f1s = []
            for _ in range(1000):
                rnd_y = np.random.permutation(y_valid)
                rnd_f1s.append(f1_score(y_valid, rnd_y))
            return np.mean(rnd_f1s)
    return acc


def get_utility_prob(x_train, y_train, x_valid, y_valid, clf, num_classes):
    """_summary_

    Args:
        x_train (_type_): _description_
        y_train (_type_): _description_
        x_valid (_type_): _description_
        y_valid (_type_): _description_
        clf (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        clf.fit(x_train, y_train)
        prob = clf.predict_proba(x_valid)
        # prob = np.amax(prob, axis=1)
        # hatY = clf.predict(x_valid)
        # acc = np.sum(prob[y_valid == hatY]) / len(y_valid)
        true_probs = prob[np.arange(len(y_valid)), y_valid]
        acc = np.mean(true_probs)
    except ValueError:
        # Training set only has a single class
        acc = 1.0 / num_classes * accuracy_score(y_valid, [y_train[0]] * len(y_valid))
    return acc


def get_utility_classwise(x_train, y_train, x_valid, y_valid, label, clf):
    devX_label = x_valid[y_valid == label]
    devY_label = y_valid[y_valid == label]
    devX_nonlabel = x_valid[y_valid != label]
    devY_nonlabel = y_valid[y_valid != label]
    # Always more than 1 class in the training set
    clf.fit(x_train, y_train)
    val_i = accuracy_score(devY_label, clf.predict(devX_label), normalize=False) / len(y_valid)
    val_i_non = accuracy_score(devY_nonlabel, clf.predict(devX_nonlabel), normalize=False) / len(y_valid)
    return [val_i, val_i_non]


def m_brier_score(y_true, y_prob, **kwargs):
    """Compute Brier score for multi-class

    Args:
        y_true (np.ndarray)
        y_prob (np.ndarray)

    Returns:
        int: Brier score

    Example:
    --------
    y_true = [0, 1, 1, 0]
    y_prob = [[0.9, 0.1],   # class 0 prob 90%, class 1 prob 10%
              [0.1, 0.9],
              [0.2, 0.8],
              [0.7, 0.3]]
    """
    sample_weight = kwargs.get("sample_weight", None)
    num_events = len(y_true)
    y_true_padding = np.zeros(shape=np.shape(y_prob))
    for i in range(num_events):
        y_true_padding[i, y_true[i]] = 1

    return np.average((y_prob - y_true_padding) ** 2,
                      weights=sample_weight)


def brier_skill_score(y_true, y_prob, y_prob_ref, **kwargs):
    """Compute the Brier skill score
    """
    sample_weight = kwargs.get("sample_weight", None)

    bsf = m_brier_score(y_true, y_prob,
                        sample_weight=sample_weight)
    bsref = m_brier_score(y_true, y_prob_ref,
                          sample_weight=sample_weight)
    return 1 - bsf / bsref


def brier_score_sqrt(y_true, y_prob, **kwargs):
    sample_weight = kwargs.get("sample_weight", None)
    res = m_brier_score(y_true, y_prob,
                        sample_weight=sample_weight)
    if res >= 1:
        return 0
    return np.sqrt(1 - res)


def weight(j, n, alpha=1.0, beta=1.0):
    log_1, log_2, log_3 = 0.0, 0.0, 0.0
    for k in range(1, j):
        log_1 += np.log(beta + k - 1)
    for k in range(1, n - j + 1):
        log_2 += np.log(alpha + k - 1)
    for k in range(1, n):
        log_3 += np.log(alpha + beta + k - 1)
    log_total = np.log(n) + log_1 + log_2 - log_3
    # print("n = {}, j = {}".format(n, j))
    log_comb = None
    if n <= 20:
        log_comb = np.log(factorial(n - 1))
    else:
        log_comb = (n - 1) * (np.log(n - 1) - 1)
    if j <= 20:
        log_comb -= np.log(factorial(j - 1))
    else:
        log_comb -= (j - 1) * (np.log(j - 1) - 1)
    if (n - j) <= 20:
        log_comb -= np.log(factorial(n - j))
    else:
        log_comb -= (n - j) * (np.log(n - j) - 1)
    # print("log_total = {}, log_comb = {}".format(log_total, log_comb))
    v = np.exp(log_comb + log_total)
    # print("v = {}".format(v))
    return v


def gr_statistic(val, t):
    v = val[:, :, 1:t + 1]
    sample_var = np.var(v, axis=2, ddof=1)  # N x K, along dimension T
    mean_sample_var = np.mean(sample_var, axis=1)  # N, along dimension K, s^2 in the paper
    sample_mean = np.mean(v, axis=2)  # N x K, along dimension T
    sample_mean_var = np.var(sample_mean, axis=1, ddof=1)  # N, along dimension K, B/n in the paper
    sigma_hat_2 = ((t - 1) * mean_sample_var) / t + sample_mean_var
    rho_hat = np.sqrt(sigma_hat_2 / (mean_sample_var + 1e-4))
    return rho_hat


def data_removal_figure(neg_lab, pos_lab, trnX, trnY, devX, devY, sorted_dct, clf_label, remove_high_value=True):
    # Create data indices for data removal
    N = trnX.shape[0]
    Idx_keep = [True] * N
    # Accuracy list
    accs = []
    if remove_high_value:
        lst = range(N)
    else:
        lst = range(N - 1, -1, -1)
    # Compute
    clf = Classifier(clf_label)
    clf.fit(trnX, trnY)
    dev = zip(devX, devY)
    dev = list(dev)
    dev_X0 = []
    dev_X1 = []
    dev_Y0 = []
    dev_Y1 = []

    for i in dev:
        if i[1] == pos_lab:
            dev_X1.append(i[0])
            dev_Y1.append(i[1])
        elif i[1] == neg_lab:
            dev_X0.append(i[0])
            dev_Y0.append(i[1])
        else:
            print(i)

    acc_0 = accuracy_score(dev_Y0, clf.predict(dev_X0), normalize=False) / len(devY)
    acc_1 = accuracy_score(dev_Y1, clf.predict(dev_X1), normalize=False) / len(devY)
    print(acc_0, acc_1)

    accs_0 = []
    accs_1 = []
    acc = accuracy_score(clf.predict(devX), devY)  # /len(devY)
    accs.append(acc)
    accs_0.append(acc_0)
    accs_1.append(acc_1)
    vals = []
    labels = []
    points = []
    ks = []
    for k in lst:
        # print(k)
        Idx_keep = [True] * N
        Idx_keep[sorted_dct[k][0]] = False
        trnX_k = trnX[Idx_keep, :]
        trnY_k = trnY[Idx_keep]
        clf = Classifier(clf_label)
        try:
            clf.fit(trnX_k, trnY_k)
            # print('trnX_k.shape = {}'.format(trnX_k.shape))
            labels.append(trnY[k])
            points.append(trnX[k])
            acc = accuracy_score(clf.predict(devX), devY)
            acc_0 = accuracy_score(dev_Y0, clf.predict(dev_X0), normalize=False) / len(devY)
            acc_1 = accuracy_score(dev_Y1, clf.predict(dev_X1), normalize=False) / len(devY)
            # print('acc = {}'.format(acc))
            ks.append(k)
            accs.append(acc)
            accs_0.append(acc_0)
            accs_1.append(acc_1)
            vals.append(sorted_dct[k][1])
        except ValueError:
            # print("Training with data from a single class")
            accs.append(0.0)
    return accs, accs_0, accs_1, vals, labels, points, ks


if __name__ == "__main__":
    dataset = "wind"
    trnX, trnY, devX, devY, tstX, tstY = get_dataset(dataset, '../../data/{}.pkl'.format(dataset), noise=False)
    print(trnY[:20])
    trnX, trnY, devX, devY, tstX, tstY = get_dataset(dataset, '../../data/{}.pkl'.format(dataset), noise=True)
    print(trnY[:20])





