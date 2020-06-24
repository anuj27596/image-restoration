from keras.layers import Input
from keras.optimizers import Adam
import keras.backend as K

import matplotlib.pyplot as plt
import numpy as np

from model import get_Model
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def mse_gr(gr):
	def loss_func(y_true, y_pred):
		return K.mean(K.square(y_pred - gr))
	return loss_func

gr_img = plt.imread('data/car.png')[:,:,:3]
M, N, C = gr_img.shape

sigma = 0.1
np.random.seed(10)
noisy_img = gr_img + sigma * np.random.randn(M, N, C)
noisy_img = np.vectorize(lambda x: 1.0 if x>1 else 0.0 if x<0 else x)(noisy_img)

F = 16

model = get_Model(Input((M, N, F)), n_filters=8, depth=4, output_channels=C)
model.compile(optimizer=Adam(), loss='mse', metrics=[mse_gr(gr_img)])

np.random.seed(2)
z = np.random.uniform(low=-1, high=1, size=(1,M,N,F))


model.fit(z, noisy_img.reshape((1,M,N,C)), epochs=2000, verbose=1)

j = model.predict(z)

plt.figure()

plt.subplot(3,1,1)
plt.imshow(noisy_img)

plt.subplot(3,1,2)
plt.imshow(j[0])

plt.subplot(3,1,3)
plt.imshow(gr_img)

plt.figure()
plt.plot(model.history.history['loss'])

plt.show()

