'''Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os, csv
from PIL import Image
import numpy as np
from scipy.stats.stats import pearsonr

### NN PARAMS:
batch_size = 512
epochs = 50
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'
rescale_factor = 24  # OK good correlation on test set with scale 24, size 1
nn_size_factor = 1 #int(24/rescale_factor)
#fr_minst = 4
width = int(1388/rescale_factor)
height = int(1040/rescale_factor)
image_dir = "/Users/apelonero/git/Telofish_NN/IMAGES/"
annotations_filename ="/Users/apelonero/git/Telofish_NN/annotations_parsed4.csv"
image_ext = "_c2.TIF"
attempt_load_model = True
test_set = []
train_set = []
max_tl = 1815.6
min_tl = 10000

###
with open(annotations_filename , "rb") as annotations_file:
    reader = csv.DictReader(annotations_file)

    for line in reader:
        out_path = image_dir+line['Folder names']+"/"+line['Folder names']+"_c2.TIF"
        line['jpeg_filename']=out_path
        line['Median']=float(line['Median'])/max_tl
        max_tl = max(max_tl, line['Median'])
        min_tl = min(min_tl, line['Median'])
        line['TL']=int(line['TL'])
        # if not os.path.isfile(out_path):
        #     filename = image_dir+line['Folder names']+"/"+line['Folder names']+"_c2.TIF"
        #     if (os.path.isfile(filename)):
        #         im = Image.open(filename)
        #         print ("Generating jpeg for %s" % filename)
        #         im.thumbnail(im.size)
        #         im.save(out_path, "JPEG", quality=100)
        #     else:
        #         print ("not found",filename)
        #         continue
        im = Image.open(out_path).resize((width, height), Image.ANTIALIAS)
        red, green, blue = im.split()
        line['jpeg'] = np.array(im).astype('float32') /255
        line['red'] = np.array(red).sum()/(width*height*255)
        if line['Set']=="Train":
            train_set.append(line)
            line['jpeg'] = np.array(im.transpose(Image.FLIP_LEFT_RIGHT)).astype('float32') /255
            train_set.append(line)
            line['jpeg'] = np.array(im.transpose(Image.FLIP_TOP_BOTTOM)).astype('float32') /255
            train_set.append(line)
            line['jpeg'] = np.array(im.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)).astype('float32') /255
            train_set.append(line)
        else:
            test_set.append(line)

print (           max_tl, min_tl, max_tl-min_tl)
x_train = np.concatenate([[line['jpeg']] for line in train_set])
x_test = np.concatenate([[line['jpeg']] for line in test_set])
y_train =  np.array([line['Median'] for line in train_set])
y_test =  np.array([line['Median'] for line in test_set])

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

### Create new test/validation with N samples
##def red_value(x, y, n_samples = 200):
##    x_2 = np.array([x[0]])
##    y_2 = np.array(x[0,:,:,0].sum())
##    for i in xrange(1,n_samples):
##        x_2 = np.append(x_2, [x[i]], axis=0)
##        y_2 = np.append(y_2, x[i,:,:,0].sum()/(width*width))
##    y_2 = y_2.astype("uint8")
##    return ((x_2, y_2))
##
##(x_train_2, y_train_2) = red_value(x_train, y_train)
##print ("computed train array")
##(x_test_2, y_test_2) = red_value(x_test, y_test)

##x_train=_2 = np.array([x_train[0]])
##y_train_2 = np.array(x_train[0,:,:,0].sum())
##for i in xrange(1,n_samples):
##    x_train_2 = np.append(x_train_2, [x_train[i]], axis=0)
##    y_train_2 = np.append(y_train_2, x_train[i,:,:,0].sum()/(width*width))
##y_train_2 = y_train_2.astype("uint8")

#raise("Break")

# Convert class vectors to binary class matrices.
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(int(32/nn_size_factor), (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(int(32/nn_size_factor), (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(int(64/nn_size_factor), (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(int(64/nn_size_factor), (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(int(512/nn_size_factor)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('linear'))

# initiate RMSprop optimizer
opt = keras.optimizers.Adam() #tf.train.AdamOptimizer(0.01) #keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='mse',
              optimizer=opt,
              metrics=['mae'])
if attempt_load_model and os.path.isfile(os.path.join(save_dir, model_name)):
    model = load_model(os.path.join(save_dir, model_name))
    print ("Model loaded")
##x_train = x_train.astype('float32')
##x_test = x_test.astype('float32')
##x_train /= 255
##x_test /= 255
##
##y_test = y_test_2 / 255
##y_train = y_train_2 / 255

##if not data_augmentation:
##    print('Not using data augmentation.')

##else:
##    print('Using real-time data augmentation.')
##    # This will do preprocessing and realtime data augmentation:
##    datagen = ImageDataGenerator(
##        featurewise_center=False,  # set input mean to 0 over the dataset
##        samplewise_center=False,  # set each sample mean to 0
##        featurewise_std_normalization=False,  # divide inputs by std of the dataset
##        samplewise_std_normalization=False,  # divide each input by its std
##        zca_whitening=False,  # apply ZCA whitening
##        zca_epsilon=1e-06,  # epsilon for ZCA whitening
##        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
##        # randomly shift images horizontally (fraction of total width)
##        width_shift_range=0.1,
##        # randomly shift images vertically (fraction of total height)
##        height_shift_range=0.1,
##        shear_range=0.,  # set range for random shear
##        zoom_range=0.,  # set range for random zoom
##        channel_shift_range=0.,  # set range for random channel shifts
##        # set mode for filling points outside the input boundaries
##        fill_mode='nearest',
##        cval=0.,  # value used for fill_mode = "constant"
##        horizontal_flip=True,  # randomly flip images
##        vertical_flip=False,  # randomly flip images
##        # set rescaling factor (applied before any other transformation)
##        rescale=None,
##        # set function that will be applied on each input
##        preprocessing_function=None,
##        # image data format, either "channels_first" or "channels_last"
##        data_format=None,
##        # fraction of images reserved for validation (strictly between 0 and 1)
##        validation_split=0.0)
##
##    # Compute quantities required for feature-wise normalization
##    # (std, mean, and principal components if ZCA whitening is applied).
##    datagen.fit(x_train)
##
##    # Fit the model on the batches generated by datagen.flow().
##    model.fit_generator(datagen.flow(x_train, y_train,
##                                     batch_size=batch_size),
##                        epochs=epochs,
##                        validation_data=(x_test, y_test),
##                        workers=4)
##
##

while(True):
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # correlation
    z_test = np.array([el[0] for el in model.predict(x_test)])*max_tl
    z_train = np.array([el[0] for el in model.predict(x_train)])*max_tl
    print ("Train", pearsonr(z_train, y_train))
    print ("Test",pearsonr(z_test, y_test))
    p_val = pearsonr(z_train, y_train)[1]
    if p_val>0.1:
        break;
    comp_array = np.vstack((np.column_stack((y_test*max_tl, z_test.round(), np.array([1]*y_test.size))),
                          np.column_stack((y_train*max_tl, z_train.round(), np.array([2]*y_train.size)))))
    np.savetxt("pred_vs_data.csv", comp_array, delimiter=",",
               header="actual,predicted,set (1=test)")


##
### Score trained model.
##scores = model.evaluate(x_test, y_test, verbose=1)
##print('Test loss:', scores[0])
##print('Test accuracy:', scores[1])
