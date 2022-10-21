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

def plot_errors(city, model, pos = -1, ind = 01, s = 0.2):
  if pos > 0:
    plt.subplot(4, 2, pos)
  maxx = np.max(ys_tests[city][:,ind])
  minn = np.min(ys_tests[city][:,ind])
  for model in models:
    name = str(model.__class__).split(".")[-1][:-2]
    ys_pred = model.predict(xs_tests[city])
    plt.scatter(ys_pred[:,ind], ys_tests[city][:,ind], alpha = 1, s = s, label = name)
    maxx = max(np.max(ys_pred[:,ind]),maxx)
    minn = min(np.min(ys_pred[:,ind]),minn)
  plt.xlabel("Predicted Energy Ouptut (Normalized)")
  plt.ylabel("Actual Energy Output (Normalized)")
  plt.legend()
  plt.title(f"{city} WEC #{ind}" if ind > 0 else f"{city} Total WECs Output")
  t = [minn, maxx]
  plt.axis(t + t)
  plt.plot(t,t,'r--')
  
plt.rcParams['figure.figsize'] = [12, 16]
i = 0

table_data = []
for city in cities:
  sequential_model = models.load_model(f"{city}_model.h5")
  linear_model = LinearRegression().fit(xs_trains[city], ys_trains[city])
  models = [linear_model, sequential_model]
  
  plot_errors(file, models, pos = i+1, ind = 0)
  plot_errors(file, models, pos = 1+2, ind = -1)
  
  table_data.append([city, get_error(city, linear_model), get_error(city, sequential_model)])
  i += 2
plt.tight_layout()
plt.show()
