from sklearn.datasets import load_boston
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
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
    def raw_data(self):
        if self.data_set == 'Boston':
            my_data = load_boston()
            df = pd.DataFrame(my_data.data, columns=my_data.feature_names)
            df['target'] = my_data.target
            return df
        # elif self.data_set == 'Wine':
        #     my_data = load_wine()
        #     df = pd.DataFrame(my_data.data, columns=my_data.feature_names)
        #     df['target'] = my_data.target
        #     return df
        else:
            return None

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
