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
		return K.mean(K.square((y_pred[:,:,:,0] - y_true[:,:,:,0]) * mask))
	return loss_func


img0 = plt.imread('data/brain_mri.png')
M, N = img0.shape

mask = np.random.uniform(size=(M,N)) > 0.5

img = img0 * mask

F = 16

model = get_Model(Input((M, N, F)), n_filters=8, depth=4, output_channels=1)
model.compile(optimizer=Adam(), loss=mse_masked(mask))

np.random.seed(2)
z = np.random.uniform(low=-1, high=1, size=(1,M,N,F))

bmcb = BMCallback()

model.fit(z, img.reshape((1,M,N,1)), epochs=2000, verbose=1, callbacks=[bmcb])
model.set_weights(bmcb.best_weights)

j = model.predict(z)

plt.set_cmap('gray')

plt.subplot(2,1,1)
plt.imshow(img)

plt.subplot(2,1,2)
plt.imshow(j[0,:,:,0])

plt.show()
