import random
import json
import pickle
import numpy
import nltk

from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.models import sequential
from tensorflow.python.keras.models import sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.optimizer_v2 import gradient_descent as GSD


def add(a, b):
    return a + b
