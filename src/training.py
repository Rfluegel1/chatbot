import random
import json
import pickle
import numpy
import nltk

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD