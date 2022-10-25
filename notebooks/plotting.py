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

# extract files
files = "Sydney Tasmania Perth Adelaide".split()
xs_tests = dict()
ys_tests = dict()
xs_trains = dict()
ys_trains = dict()
xs_valids = dict()
ys_valids = dict()

for file in files:
  csvfile = open(f"/notebooks/WECs_dataset/{file}_Data.csv")
  reader = csv.reader(csvfile, delimiter = ",")
  rows = [[float(v) for v in row] for row in reader]
  rows = no.asarray(rows, dtype = "float32")
  xs = rows[:,0:32]
  ys = rows[:,32:49]
  
  # standardize xs and ys
  ys = ys / np.max(ys)
  xs = xs / np.max(xs)
  
  xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size = 0.2, random_state = )
  xs_trains[file] = xs_train
  ys_trains[file] = ys_train
  xs_tests[file] = xs_test
  ys_tests[file] = ys_test
  with open(f"/notebooks/models/{file}_hist.pickle","rb") as f:
    p = pickle.load(f)
  xs_valids[file] = p.validation_data[0]
  ys_valids[file] = p.validation_data[1]

def get_error(file, model):
  """ Returns MAPE by calculating ys_pred from xs_test array and comparing to ys_test array"""
  ys_pred = model.predcit(xs_tests[file])
  mape = tf.keras.losses.MeanAbsolutePercentageError()
  error = mape(ys_pred, ys_tests[file]).numpy()
  return error

def plot_errors(city, models, pos, s = 0.2):
  for ind in range(0,17):
    plt.subplot(4, 17, pos)
    for model in models:
      name = str(model.__class__).split(".")[-1][:-2]
      ys_pred = model.predict(xs_test[city])
      plt.scatter(ys_pred[:,ind], ys_test[city][:,ind], alpha = 1, s = s, label = name)
      maxx = np.max(ys_pred[:,ind])
      minn = np.min(ys_pred[:,ind])
    plt.legend()
    plt.title(f"{city} WEC #{ind}" if ind <= 16 else f"{city} Total WECs Output")
    plt.axis(t + t)t
    plt.plot(t,t,'r--')
  plt.xlabel("Predicted Energy Ouptut (Normalized)")
  plt.ylabel("Actual Energy Output (Normalized)")
  
plt.rcParams['figure.figsize'] = [12, 16]
i = 0

table_data = []
for city in cities:
  sequential_model = models.load_model(f"{city}_model.h5")
  linear_model = LinearRegression().fit(xs_train[city], ys_train[city])
  models = [linear_model, sequential_model]
  
  plot_errors(file, models, pos = i+1, ind = 0)
  plot_errors(file, models, pos = 1+2, ind = -1)
  
  table_data.append([city, get_error(city, linear_model), get_error(city, sequential_model)])
  i += 2
plt.tight_layout()
plt.show()
