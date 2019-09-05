# from keras.models import load_model
# from keras.utils import plot_model
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
#
# import os # to handle keras AVX/FMA warning on non-supported platforms
#
# ### Load model:
# model = load_model('/Users/apelonero/git/Telofish_NN/saved_models/keras_cifar10_trained_model.h5')
#
# # Just disables the  warning, doesn't enable AVX/FMA
# # Comment out if on Nvidia GPU
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# ### Plots:
# plot_model(model, to_file='model.png')
#
# #SVG(model_to_dot(model).create(prog='dot', format='svg'))

#~~~~~~~
# Visualize training history

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy
import os # to handle keras AVX/FMA warning on non-supported platforms

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

model = load_model('/Users/apelonero/git/Telofish_NN/saved_models/keras_cifar10_trained_model.h5')

# Just disables the  warning, doesn't enable AVX/FMA
# Comment out if on Nvidia GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Fit the model
history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()