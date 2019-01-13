import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from get_data import MasterData
from get_data import get_boston
from sklearn.preprocessing import MinMaxScaler

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
model = tf.estimator.DNNRegressor(hidden_units=[6, 6, 6], feature_columns=feat_cols)

# Train model
model.train(input_fn=input_function, steps=25000)