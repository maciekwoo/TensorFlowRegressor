import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from get_data import MasterData
from get_data import my_grid_search
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

md = MasterData('Boston')
scaler = MinMaxScaler()

# Scale the data
X_train = pd.DataFrame(data=scaler.fit_transform(md.x_train), columns=md.x_train.columns, index=md.x_train.index)
X_test = pd.DataFrame(data=scaler.fit_transform(md.x_eval), columns=md.x_eval.columns, index=md.x_eval.index)

model = MLPRegressor(hidden_layer_sizes=(13, 13, 13), activation='logistic', alpha=0.0001, solver='lbfgs')

# do grid search on model
# my_grid_search(model,X_train, md.y_train)

model.fit(X_train, md.y_train)

predictions = model.predict(X_test)

pred_df = pd.DataFrame(data=predictions)

output = mean_squared_error(md.y_eval, pred_df) ** 0.5

comparison = pd.concat([md.y_eval_reset, pred_df], axis=1)
comparison.columns = ['original_y', 'predicted_target']
comparison['diff'] = comparison['original_y'] - comparison['predicted_target']
