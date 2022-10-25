import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import pickle

from tensorflow.keras.losses import MeanAbsolutePercentageError as MAPE
from random import choice, random
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from IPython.display import display
from sklearn.model_selection import LinearRegression
from IPython.core.display import display, HTML
from tqdm.keras import TqdmCallback

# check if the nvidia-docker container supplied GPU and if tensorflow is using it
tf.config.experimental.list_physical_devices('GPU')

# extract files
cities = "Sydney Tasmania Perth Adelaide".split()
xs_tests = dict()
ys_tests = dict()
xs_trains = dict()
ys_trains = dict()

for city in cities:
  csvfile = open(f"/notebooks/WECs_DataSet/{city}_Data.csv")
  reader = csv.reader(csvfile, delimiter = ",")
  rows = [[float(v) for v in row] for row in reader]
  rows = no.asarray(rows, dtype = "float32")
  xs = rows[:,0:32]
  ys = rows[:,32:49]
  
  # standardize xs and ys
  ys = ys / np.max(ys)
  xs = xs / np.max(xs)
  
  xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size = 0.2, random_state = 100)
  xs_trains[city] = xs_train
  ys_trains[city] = ys_train
  xs_tests[city] = xs_test
  ys_tests[city] = ys_test

# MAPE
def get_error(city, model):
  """ Returns MAPE by calculating ys_pred from xs_test array and comparing to ys_test array"""
  ys_pred = model.predict(xs_tests[city])
  mape = tf.keras.losses.MeanAbsolutePercentageError()
  error = mape(ys_pred, ys_tests[city]).numpy()
  return error

# model training
def train_model(city, patience = 100, verbose = 0):
  model = models.Sequential()
  model.add(layers.Dense(1024, input_dim = 32, activation = "relu"))
  model.add(layers.Dropout(0.1))
  model.add(layers.Dense(2048, activation = "relu"))
  model.add(layers.Dense(1024, activation = "relu"))
  model.add(layers.Dense(17, activation = 'linear'))
    
  earlyStopping = EarlyStopping(monitor = 'val_loss',
                                patience = patience,
                                verbose = verbose,
                                mode = 'min')
  mcp_save = ModelCheckpoint('tmp.h5',
                             save_best_only = True,
                             monitor = 'val_loss',
                             mode = 'min')
  reduce_lr_loss = ReduceLROnPlateau(monitor = 'val_loss',
                                     factor = 0.2,
                                     patience = 70, 
                                     verbose = 1,
                                     min_delta = 1e-5,
                                     mode = 'min')
  logger = TqdmCallback(verbose = verbose)
    
  model.compile("adam",loss = "mean_absolute_percentage_error")
  loss_hist = model.fit(xs_trains[city],
                        ys_trains[city],
                        epochs = 15000,
                        shuffle = True,
                        verbose = 0,
                        validation_data = (xs_tests[city], ys_tests[city]), 
                        batch_size = 2048,
                        callbacks = [earlyStopping, mcp_save, reduce_lr_loss,logger])
  model.load_weights('tmp.h5')
  os.remove('tmp.h5')
  return model, loss_hist
  
for city in cities:
  print(city)
  model, loss_hist = train_model(city,
                                 patience = 100,
                                 verbose = 1)
  model.save(f"{city}_model.h5")
  with open(f"{city}_hist.pickle", "wb") as f:
    pickle.dump(loss_hist, f)
  linear_model = LinearRegression().fit(xs_trains[city], ys_trains[city])
  error1 = get_error(city, linear_model)
  error2 = get_error(city, model)
  print("Linear Model MAPE: \t error1:.3f")
  print("Sequential Model MAPE: \t error2:.3f")
