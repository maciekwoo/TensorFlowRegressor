import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from get_data import MasterData
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

md = MasterData('Boston')
scaler = MinMaxScaler()

# Scale the data
X_train = pd.DataFrame(data=scaler.fit_transform(md.x_train), columns=md.x_train.columns, index=md.x_train.index)
X_test = pd.DataFrame(data=scaler.fit_transform(md.x_eval), columns=md.x_eval.columns, index=md.x_eval.index)

# Create feature columns
feat_cols = md.x_train.columns.tolist()
my_features = []

for col in feat_cols:
    my_features.append(tf.feature_column.numeric_column(col))

# Create input function
input_function = tf.estimator.inputs.pandas_input_fn(x=X_train, y=md.y_train, batch_size=10, num_epochs=1000,
                                                     shuffle=True)

# Create the estimator model
model = tf.estimator.DNNRegressor(hidden_units=[13, 13, 13], feature_columns=my_features)

# Train model
model.train(input_fn=input_function, steps=5000)

# Create predictor input function
predictor_input_function = tf.estimator.inputs.pandas_input_fn(
    x=X_test,
    batch_size=300,
    num_epochs=1,
    shuffle=False)

prediction_gen = model.predict(predictor_input_function)

predictions = list(prediction_gen)

# Flatten output of tf regressor
prediction_values = []
for prediction in predictions:
    prediction_values.append(prediction['predictions'])

output = mean_squared_error(md.y_eval, prediction_values) ** 0.5

# Flatten even further to list of values....
out_flattened = pd.Series(np.concatenate(prediction_values).ravel().tolist())

# I want to display the original values and results side by side
out_flattened.reset_index(drop=True, inplace=True)
original_y = md.y_eval
original_y.reset_index(drop=True, inplace=True)

comparison = pd.concat([original_y, out_flattened], axis=1)
comparison.columns = ['original_y', 'predicted_target']
comparison['diff'] = comparison['original_y'] - comparison['predicted_target']
