
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
import concurrent.futures
from .util import get_model
## Local module: load data valuation methods
from .loo import loo
from .P_Shapley import truncated_mc
from .beta_shapley import beta_shapley
from .params import Parameters # Import default model parameters


class DataValuation(object):
    def __init__(self, trnX, trnY, devX=None, devY=None):
        '''
        trn_X, trn_Y - Input/output for training, also the examples for
                       being valued
        val_X, val_Y - Input/output for validation, also the examples used
                       for estimating the values of (trn_X, trn_Y)
        '''
        self.trnX, self.trnY = trnX, trnY
        if devX is None:
            self.valX, self.devY = trnX, trnY
        else:
            self.devX, self.devY = devX, devY
        self.values = {} # A rank list of
        self.clf = None # instance of classifier
        params = Parameters()
        self.params = params.get_values()


    def estimate(self, clf=None, method='loo', dataset = None, params=None):
        '''
        clf - a classifier instance (Logistic regression, by default)
        method - the data valuation method (LOO, by default)
        params - hyper-parameters for data valuation methods
        '''
        self.values = {}
        if clf is None or clf == 'LR':
            self.clf = LR(solver="liblinear", max_iter=500, random_state=0)
        else:
            self.clf = get_model(clf, method, dataset)

        if params is not None:
            print("Overload the model parameters with the user specified ones: {}".format(params))
            self.params = params

        # Call data valuation functions
        if method == 'loo':
            # Leave-one-out
            vals = loo(self.trnX, self.trnY, self.devX, self.devY, self.clf)
            for idx in range(len(vals)):
                self.values[idx] = vals[idx]
        elif method == 'tmc-shapley':
            n_iter = self.params['tmc_iter']
            tmc_thresh = self.params['tmc_thresh']
            #
            valsdict = truncated_mc(self.trnX, self.trnY, self.devX, self.devY,
                                   self.clf, n_iter, tmc_thresh)
            # for method in len(vals):
            #     for idx in range(len(method)):
            #         self.values[idx] = vals[method][idx]
            #     self.valuesList.append(self.values)
            return valsdict

        elif method == 'beta-shapley':
            # Beta Shapley
            n_iter = self.params['beta_iter']
            alpha, beta = self.params['alpha'], self.params['beta']
            rho = self.params['rho']
            n_chain = self.params['beta_chain']
            vals = beta_shapley(self.trnX, self.trnY, self.devX, self.devY,
                                    self.clf, alpha, beta, rho, n_chain, n_iter)
            for idx in range(len(vals)):
                self.values[idx] = vals[idx]
        elif method == 'inf-func':
            n_iter = self.params['if_iter']
            second_order_grad = self.params['second_order_grad']
            for_high_value = self.params['for_high_value']
            vals = inf_func(self.trnX, self.trnY, self.devX, self.devY,
                                clf=self.clf,
                                second_order_grad=second_order_grad,
                                for_high_value=for_high_value)
            for idx in range(len(vals)):
                self.values[idx] = vals[idx]
        else:
            raise ValueError("Unrecognized data valuation method: {}".format(method))
        return self.values

    def get_values(self):
        '''
        return the data values
        '''
        if self.values is not None:
            return self.values
        else:
            raise ValueError("No values computed")
