from keras.layers import *
from keras.models import Model
import tensorflow as tf

def conv2d_block(input_tensor, n_filters, kernel_size=3):

	x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding="same")(input_tensor)
	x = Activation("softplus")(x)

	x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding="same")(x)
	x = Activation("softplus")(x)

	return x

def get_Model(input_img, n_filters=16, depth=4, res_exp=0, output_channels=3):
	tf.random.set_seed(2)

	l = conv2d_block(input_img, n_filters=n_filters, kernel_size=3)
	l = MaxPooling2D((2, 2)) (l)

	for i in range(1,depth):
		l = conv2d_block(l, n_filters=n_filters * 2**i, kernel_size=3)
		l = MaxPooling2D((2, 2)) (l)

	l = conv2d_block(l, n_filters=n_filters * 2**depth, kernel_size=3)

	for i in range(depth+res_exp):
		l = Conv2DTranspose(int(n_filters * 2**(depth-i-1)), (3, 3), strides=(2, 2), padding='same') (l)
		l = conv2d_block(l, n_filters=int(n_filters * 2**(depth-i-1)), kernel_size=3)

	
	outputs = Conv2D(output_channels, (1, 1), activation='sigmoid') (l)
	model = Model(inputs=[input_img], outputs=[outputs])

	return model
