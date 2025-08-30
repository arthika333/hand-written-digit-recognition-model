import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
(xtrain,ytrain),(xtest,ytest)=mnist.load_data()

classnames=[str(i) for i in range(10)]
classnames

#normalizing the values
tf.reduce_max(xtrain),tf.reduce_min(xtrain)

#rescaling the values
xtrainnormal=xtrain/255.
xtestnormal=xtest/255.

#creating the model
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(42)
model=Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128,activation='relu'),
    layers.Dense(128,activation='relu'),
    layers.Dense(10,activation='softmax')
])

model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
     
history=model.fit(xtrainnormal,ytrain,
                  epochs=20,
                  shuffle=True)

results=model.evaluate(xtestnormal,ytest)
results

#saving the model
model.save('savedmodel.keras')
savedmodel=tf.keras.models.load_model('/content/savedmodel.keras')
