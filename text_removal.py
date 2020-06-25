from keras.layers import Input
from keras.callbacks import Callback
from keras.optimizers import *
import keras.backend as K

import matplotlib.pyplot as plt
import numpy as np

from model import get_Model
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class BMCallback(Callback):
	def __init__(self):
		self.best_weights = None
		self.min_loss = 1

	def on_epoch_end(self, epoch, logs=None):
		if logs['loss'] < self.min_loss:
			self.min_loss = logs['loss']
			self.best_weights = self.model.get_weights()

def mse_masked(mask):
	def loss_func(y_true, y_pred):
		return K.mean(K.square((y_pred - y_true) * mask))
	return loss_func


img = plt.imread('data/cat.png')[:,:,:3]
M, N, C = img.shape

mask = (np.sum(img, axis=2) > 0).reshape(M, N, 1)	# pixels of text are (0,0,0) (black)

F = 16

model = get_Model(Input((M, N, F)), n_filters=16, depth=4, output_channels=C)
model.compile(optimizer=Adam(), loss=mse_masked(mask))

np.random.seed(1)
z = np.random.uniform(low=-1, high=1, size=(1,M,N,F))

bmcb = BMCallback()

model.fit(z, img.reshape((1,M,N,C)), epochs=2000, verbose=1, callbacks=[bmcb])
model.set_weights(bmcb.best_weights)

j = model.predict(z)

plt.figure()

plt.subplot(2,1,1)
plt.imshow(img)

plt.subplot(2,1,2)
plt.imshow(j[0])

plt.figure()
plt.plot(model.history.history['loss'])

plt.show()
