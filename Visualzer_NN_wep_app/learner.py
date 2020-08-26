import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


plt.figure(figsize=(10, 10))
for i in range(0, 16):
	plt.subplot(4, 4, i + 1)
	plt.imshow(x_train[i], cmap = 'binary')
	plt.xlabel(str(y_train[i]))
	plt.xticks([])
	plt.yticks([])

plt.show()


x_train =  np.reshape(x_train, (60000, 28 * 28))
x_test = np.reshape(x_test, (10000, 28 * 28))

# make data between zero and one instead of zero to 255
x_train = x_train / 255
x_test = x_test / 255


model = tf.keras.Sequential([
	tf.keras.layers.Dense(32, activation='sigmoid'), 
	tf.keras.layers.Dense(32, activation='sigmoid'), 
	tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
	loss = 'sparse_categorical_crossentropy', 
	optimizer='adam', 
	metrics=['accuracy']
)


# train the model 

_ = model.fit(x_train, y_train, 
		validation_data = (x_test, y_test), 
		epochs=20, batch_size=512, 
		verbose = 2
	)

# save the model 

model.save('model.h5')