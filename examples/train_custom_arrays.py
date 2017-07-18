"""
DESCRIPTION
"""
import sys
sys.path.append('/Users/mohammed/github/neural-vision/src/')

import numpy as np 

import cPickle as pickle
from sklearn.model_selection import train_test_split
from train import train_models_array as TMA
from postprocessing import TestSetAnalysis as TSA

#==============================================================================

# Parameters:
#------------

xname = '/Users/mohammed/Dropbox/deeplensing/Data/Simulation/SLChallenge/imadjust.npy'
yname = '/Users/mohammed/Dropbox/deeplensing/Data/Simulation/SLChallenge/classification.npy'
save_dir='/Users/mohammed/Desktop/testing/'
rescale = 1.0/255.0

nlayers_conv=3
filters_conv=[64, 128, 256]
nlayers_dense=2
filters_dense=[1024, 1024]
conv_kernel=(3,3)
pooling_kernel=(2,2)
val_split=0.2
batch_size=32
activation='relu'
loss='categorical_crossentropy'
metrics=['accuracy']
optimizer='adadelta'
nb_epoch=1
verbose=1
save=True
savefilename='mymodel.hdf5'
print_metadata=True
seed = 42

#==============================================================================

def to_categorical(y, num_classes):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


xdata = np.load(xname)
ydata = np.load(yname)

xdata = np.swapaxes(xdata, 1,3)
xdata = np.swapaxes(xdata, 1,2)
ydata = to_categorical(ydata, 2)

X_train, X_test, y_train, y_test = train_test_split(xdata, ydata, test_size=val_split, random_state=seed)

#==============================================================================

ob = TMA(X_train, y_train, X_test, y_test, save_dir, rescale)

hist = ob.train_custom_model(nlayers_conv, filters_conv, \
                    nlayers_dense, filters_dense, \
                    conv_kernel, pooling_kernel, \
                    val_split, batch_size,\
                    activation, loss, metrics, optimizer, \
                    nb_epoch, verbose, \
                    save, savefilename, \
                    print_metadata)

#==============================================================================

obj = TSA(save_dir+savefilename, False)

obj.predict_array(X_train, y_train, batchsize=batch_size, rescale=rescale)
mydict = obj.get_information_dictionary()
pickle.dump(mydict, \
	open(save_dir+savefilename.replace('.hdf5', '_postprocess_train.pkl'), 'w'), -1)

obj.predict_array(X_test, y_test, batchsize=batch_size, rescale=rescale)
mydict = obj.get_information_dictionary()
pickle.dump(mydict, \
    open(save_dir+savefilename.replace('.hdf5', '_postprocess_valid.pkl'), 'w'), -1)

pickle.dump(hist.history, open(save_dir+savefilename.replace('.hdf5', '_hist.p'), "w"))

#==============================================================================