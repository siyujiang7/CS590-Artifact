from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.05,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest')
for index in xrange(1,18):
	img = load_img('train_test_data/'+str(index)+'.png')  # this is a PIL image
	x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
	x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

	# the .flow() command below generates batches of randomly transformed images
	# and saves the results to the `preview/` directory
	i = 0
	for batch in datagen.flow(x, batch_size=1,
	                          save_to_dir='preview', save_prefix=str(index), save_format='jpg'):
	    i += 1
	    if i > 19:
	        break  # otherwise the generator would loop indefinitely

