from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Multiply, GlobalMaxPooling2D, MaxPool2D, Flatten, concatenate, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from tensorflow import keras
import tensorflow as tf
from dataset_load import *
from losses import *
from plot import *
import numpy as np
from tensorflow.python.ops import math_ops
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import array
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as ptl
from tensorflow.keras.models import Model
import datetime
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

inp = 128    

def Dual_Proposed(x_train, x_test, y_train, y_test, numberofmodels, epoch, batch_size):
    
    model_name = "Dual_stream"
    
    inputs = keras.Input(shape=(inp, inp, 3))
    
    CNN1 = Conv2D(filters=4, kernel_size=5,strides=1, padding='same', activation='relu')(inputs)

    CNN2 = Conv2D(filters=8, kernel_size=3,strides=1, padding='same', activation='relu')(CNN1)

    Skip1=concatenate([CNN1, CNN2])

    CNN3 = Conv2D(filters=16, kernel_size=1,strides=1, padding='same', activation='relu')(Skip1)

    Skip2=concatenate([Skip1, CNN3])

    CNN = MaxPooling2D(pool_size=2)(Skip2)

    CNN = Dense(32)(CNN)

    CNN = Flatten(name='CNN')(CNN)

    CNN = Dense(32)(CNN)

    plain_CNN = Conv2D(filters=4, kernel_size=5,strides=1, padding='same', activation='relu')(inputs)

    plain_CNN= Conv2D(filters=8, kernel_size=3,strides=1, padding='same', activation='relu')(plain_CNN)

    plain_CNN= MaxPooling2D(pool_size=2)(Skip2)

    plain_CNN= Dense(32)(plain_CNN)

    plain_CNN= Flatten(name='CNN')(plain_CNN)

    plain_CNN= Dense(7,  activation='softmax')(plain_CNN)

    model = keras.Model(inputs, plain_CNN, name = "DualStreamNetwork")


    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, verbose=1, validation_data=(x_test, y_test)) 
    
    model.save("trained_Model/Model.h5")
    
    y_pred = model.predict(x_test, verbose=1, batch_size=batch_size)
    
    y_pred=np.argmax(y_pred, axis=1)
    
    my_plot(history, model_name)
    
    losses_function(y_test, y_pred, model_name, numberofmodels, model, x_test)



    
