from sklearn.datasets import load_boston
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd


# define function to get boston data
# this is for testing purposes - seeing issues with debugger in pycharm
def get_boston():
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['target'] = boston.target
    return df


class MasterData:
    """
    Object with boston data as attribute
    """

    def __init__(self, data_set):
        self.data_set = data_set

    @property
    def raw_data(self) -> pd.DataFrame:
        if self.data_set == 'Boston':
            my_data = load_boston()
            df = pd.DataFrame(my_data.data, columns=my_data.feature_names)
            df['target'] = my_data.target
            return df
        elif self.data_set == 'Wine':
            my_data = load_wine()
            df = pd.DataFrame(my_data.data, columns=my_data.feature_names)
            df['target'] = my_data.target
            return df
        else:
            return pd.DataFrame(data=[0, 1], index=[1, 1], columns=['data', 'target'])

    @property
    def train_test(self) -> list:
        x_data = self.raw_data.drop(['target'], axis=1)
        y_true = self.raw_data['target']
        split_data = train_test_split(x_data, y_true, test_size=0.3, random_state=101)
        return split_data

    @property
    def x_train(self) -> pd.DataFrame:
        x_train = self.train_test[0]
        return x_train

    @property
    def x_eval(self) -> pd.DataFrame:
        x_eval = self.train_test[1]
        return x_eval

    @property
    def y_train(self) -> pd.DataFrame:
        y_train = self.train_test[2]
        return y_train

    @property
    def y_eval(self) -> pd.DataFrame:
        y_eval = self.train_test[3]
        return y_eval

    @property
    def y_eval_reset(self) -> pd.DataFrame:
        y_eval_reset = self.y_eval
        y_eval_reset = y_eval_reset.reset_index()
        y_eval_reset.drop('index', axis=1, inplace=True)
        return y_eval_reset


def my_grid_search(model, x_data, y_data):
    """
    :type y_data: pd.DataFrame
    :type x_data: pd.DataFrame
    :param model: 'scikit learn model'
    """
    # grid search function
    hidden_layer_sizes = [(6, 6, 6), (13, 13, 13), (25, 25, 25)]
    activation = ['logistic', 'tanh', 'relu']
    alpha = [0.0001, 0.001, 0.01]
    solver = ['lbfgs', 'sgd', 'adam']

    param_grid = dict(hidden_layer_sizes=hidden_layer_sizes, activation=activation, alpha=alpha, solver=solver)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(x_data, y_data)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
