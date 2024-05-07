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


def read_json(filename):
    file = open(filename)
    json_dict = json.loads(file.read())
    file.close()
    return json_dict
