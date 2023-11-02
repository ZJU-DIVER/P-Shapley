import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from sacred import Experiment

ex = Experiment('psv_motivation')


@ex.config
def cfg():
    problem = "classification"
    dataset = "wind"


# ============================================================================
# prepare data
# ============================================================================
def make_balance_sample(data, target):
    """
    Balance binary classification dataset by oversampling the minority class
    (only for binary classification datasets).

    Args:
        data: array-like, feature matrix
        target: array-like, target vector

    Returns:
        Tuple containing balanced feature matrix and balanced target vector
    """
    p = np.mean(target)
    if p == 0.5:
        return data, target

    minor_class = 1 if p < 0.5 else 0

    index_minor_class = np.where(target == minor_class)[0]
    n_minor_class = len(index_minor_class)
    n_major_class = len(target) - n_minor_class
    new_minor = np.random.choice(index_minor_class,
                                 size=n_major_class - n_minor_class,
                                 replace=True)

    data = np.concatenate([data, data[new_minor]])
    target = np.concatenate([target, target[new_minor]])
    return data, target


@ex.capture
def load_data(problem, dataset, **kwargs):
    """
    Load the dataset for the given problem and dataset.

    Args:
        problem (str): Type of problem, e.g., 'classification'.
        dataset (str): Name of the dataset to load.
        kwargs (dict): Additional arguments.

    Returns:
        Tuple containing training, validation, and test sets, and flipped indexes in training set.

    Note:
    - (X,y): data to be valued.
    - (X_val, y_val): data to be used for evaluation.
    - (X_test, y_test): data to be used for point removal/addition experiments (same distribution as validation).
    """
    if problem == 'classification':
        n_data_to_be_valued = kwargs.get('n_data_to_be_valued', 200)
        n_val = kwargs.get('n_val', 200)
        n_test = kwargs.get('n_test', 1000)
        clf_path = kwargs.get('clf_path', '.')

        (X, y), (X_val, y_val), (X_test, y_test) = \
            load_classification_dataset(n_data_to_be_valued=n_data_to_be_valued,
                                        n_val=n_val,
                                        n_test=n_test,
                                        dataset=dataset,
                                        clf_path=clf_path)
        # training is flipped
        f1 = np.random.choice(np.where(y == 0)[0], n_data_to_be_valued // 10 // 2, replace=False)
        f2 = np.random.choice(np.where(y == 1)[0], n_data_to_be_valued // 10 // 2, replace=False)
        flipped_index = np.concatenate((f1, f2), axis=None)
        y[flipped_index] = (1 - y[flipped_index])

        # validation is also flipped
        f1 = np.random.choice(np.where(y_val == 0)[0], n_val // 10 // 2, replace=False)
        f2 = np.random.choice(np.where(y_val == 1)[0], n_val // 10 // 2, replace=False)
        flipped_val_index = np.concatenate((f1, f2), axis=None)
        y_val[flipped_val_index] = (1 - y_val[flipped_val_index])
        return (X, y), (X_val, y_val), (X_test, y_test), flipped_index
    else:
        raise NotImplementedError('Check problem')


def load_classification_dataset(n_data_to_be_valued=200,
                                n_val=100,
                                n_test=1000,
                                dataset='gaussian',
                                clf_path='.'):
    """
    Load classification datasets for point addition experiments.

    Args:
        n_data_to_be_valued (int): Number of data points to be valued.
        n_val (int): Number of data points for utility function evaluation.
        n_test (int): Number of data points for point addition experiment evaluation.
        dataset (str): Name of the dataset to load.
        clf_path (str): Path to classification datasets.

    Returns:
        Tuple containing training, validation, and test data.
    """
    if dataset == 'gaussian':
        print('-' * 50)
        print('GAUSSIAN-C')
        print('-' * 50)
        n, input_dim = 50000, 5
        data = np.random.normal(size=(n, input_dim))
        beta_true = np.array([2.0, 1.0, 0.0, 0.0, 0.0]).reshape(input_dim, 1)
        p_true = np.exp(data.dot(beta_true)) / (1. + np.exp(data.dot(beta_true)))
        target = np.random.binomial(n=1, p=p_true).reshape(-1)
    elif dataset == 'covertype':
        print('-' * 50)
        print('Covertype')
        print('-' * 50)
        # (581012, 54)
        df = fetch_openml(name="covertype", version=2, cache=True, data_home=clf_path)
        data, target = df.data.toarray(), df.target
        target = ((target == '1') + 0.0).astype(int)
    elif dataset == 'phoneme':
        print('-' * 50)
        print('Phoneme')
        print('-' * 50)
        df = fetch_openml(name="phoneme", version=1, as_frame=True, cache=True, data_home=clf_path)
        data, target = df.data.values, df.target.values
        target = ((target == '1') + 0.0).astype(int)
    elif dataset == 'wind':
        print('-' * 50)
        print('Wind')
        print('-' * 50)
        df = fetch_openml(name="wind", version=2, as_frame=True, cache=True, data_home=clf_path)
        data, target = df.data.values, df.target.values
        target = ((target == 'P') + 0.0).astype(int)
    elif dataset == '2dplanes':
        print('-' * 50)
        print('2DPlanes')
        print('-' * 50)
        df = fetch_openml(name="2dplanes", version=2, as_frame=True, cache=True, data_home=clf_path)
        data, target = df.data.values, df.target.values
        target = ((target == 'P') + 0.0).astype(int)
    else:
        assert False, f"Check {dataset}"

    data, target = make_balance_sample(data, target)
    # Get a permutation
    idxes = np.random.permutation(len(data))
    data, target = data[idxes], target[idxes]

    # Split the data
    X_temp, X_test, y_temp, y_test = \
        train_test_split(data, target, test_size=n_test, stratify=target, random_state=42)
    X, X_val, y, y_val = \
        train_test_split(X_temp, y_temp, test_size=n_val, stratify=y_temp, random_state=42)
    _, X, _, y = \
        train_test_split(X, y, test_size=n_data_to_be_valued, stratify=y, random_state=42)
    # Normalize the data
    X_mean, X_std = np.mean(X, 0), np.std(X, 0)

    def normalizer_fn(x):
        return (x - X_mean) / np.clip(X_std, 1e-12, None)

    X, X_val, X_test = normalizer_fn(X), normalizer_fn(X_val), normalizer_fn(X_test)
    return (X, y), (X_val, y_val), (X_test, y_test)


# ============================================================================
# compute the value
# ============================================================================
