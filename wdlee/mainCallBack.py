import tensorflow as tf
import pandas as pd
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt


print(tf.__version__)

(trainX, trainY), (testX, testY) = boston_housing.load_data()

print(len(trainX))
print(len(trainY))

print(trainX[0])
print(trainY[0])

meanX = trainX.mean()
stdX = trainX.std()

trainX -= meanX
trainX /= stdX

testX -= meanX
testX /= stdX

meanY = trainY.mean()
stdY = trainY.std()

trainY -= meanY
trainY /= stdY

testY -= meanY
testY /= stdY

print(trainX[0])
print(trainY[0])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=52,activation='relu',input_shape=(13,)),
    tf.keras.layers.Dense(units=39,activation='relu'),
    tf.keras.layers.Dense(units=26,activation='relu'),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.07), loss='mse')
model.summary()

history = model.fit(trainX,trainY, epochs=300, batch_size=32,validation_split=0.25, callbacks=[tf.keras.callbacks.EarlyStopping(patience=3,monitor='val_loss')])


#plt.plot(history.history['loss'], 'b-', label='loss')
#plt.plot(history.history['val_loss'], 'r--', label='val_loss')
#plt.xlabel('epoch')
#plt.legend()
#plt.show()

eval = model.evaluate(testX, testY)

print(eval)

predY = model.predict(testX)

plt.plot(testY,predY,'b.')
plt.axis([min(testY), max(testY), min(testY), max(testY)])

plt.plot([min(testY), max(testY)] , [min(testY), max(testY)] , ls = "--", c= "0.3")
plt.xlabel('testY')
plt.xlabel('predY')
plt.show()