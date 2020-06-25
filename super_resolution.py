from keras.layers import Input
from keras.optimizers import Adam
from keras.callbacks import Callback
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


def scaled_loss(scale_exp):
	def loss_func(y_true, y_pred):
		s = 2**scale_exp
		down_ypred = y_pred[:,::s,::s] * 0
		for i in range(s):
			for j in range(s):
				down_ypred = down_ypred + y_pred[:, i::s, j::s]
		down_ypred = down_ypred / s**2

		mse = K.mean(K.square(y_true - down_ypred))
		var = K.mean(K.square(y_pred[:,1:,:] - y_pred[:,:-1,:])) + K.mean(K.square(y_pred[:,:,1:] - y_pred[:,:,:-1]))
		return mse + var/12
	return loss_func

img = np.zeros((56,48))
img[2:-2,2:-2] = plt.imread('data/mri_lowres.png')
M, N = img.shape

F = 16


model = get_Model(Input((M, N, F)), n_filters=16, depth=3, res_exp=2, output_channels=1)
model.compile(optimizer=Adam(), loss=scaled_loss(2))

np.random.seed(2)
z = np.random.uniform(low=-1, high=1, size=(1,M,N,F))

bmcb = BMCallback()

model.fit(z, img.reshape((1,M,N,1)), epochs=3000, verbose=1, callbacks=[bmcb])
model.set_weights(bmcb.best_weights)

j = model.predict(z)

plt.set_cmap('gray')

plt.subplot(2,1,1)
from skimage.transform import resize
plt.imshow(resize(img, (4*M, 4*N)))

plt.subplot(2,1,2)
plt.imshow(j[0,:,:,0])

plt.figure()
plt.plot(model.history.history['loss'])

plt.show()

