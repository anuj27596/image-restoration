from keras.layers import *
from keras.models import Model

def conv2d_block(input_tensor, n_filters, kernel_size=3):

	x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding="same")(input_tensor)
	x = Activation("softplus")(x)

	x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding="same")(x)
	x = Activation("softplus")(x)

	return x

def get_Model(input_img, n_filters=16, depth=4, output_channels=3):

	l = conv2d_block(input_img, n_filters=n_filters, kernel_size=3)
	l = MaxPooling2D((2, 2)) (l)

	for i in range(1,depth):
		l = conv2d_block(l, n_filters=n_filters * 2**i, kernel_size=3)
		l = MaxPooling2D((2, 2)) (l)

	l = conv2d_block(l, n_filters=n_filters * 2**depth, kernel_size=3)

	for i in range(depth):
		l = Conv2DTranspose(n_filters * 2**(depth-i-1), (3, 3), strides=(2, 2), padding='same') (l)
		l = conv2d_block(l, n_filters=n_filters*8, kernel_size=3)

	# c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3)
	# p1 = MaxPooling2D((2, 2)) (c1)

	# c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3)
	# p2 = MaxPooling2D((2, 2)) (c2)

	# c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3)
	# p3 = MaxPooling2D((2, 2)) (c3)

	# c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3)
	# p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
	
	# c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3)
	
	# u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
	# c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3)

	# u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
	# c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3)

	# u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
	# c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3)

	# u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
	# c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3)
	
	outputs = Conv2D(output_channels, (1, 1), activation='sigmoid') (l)
	model = Model(inputs=[input_img], outputs=[outputs])

	return model
