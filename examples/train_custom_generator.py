"""
DESCRIPTION
"""
import sys
sys.path.append('/Users/irshad/github/neural-vision/src/')

# import cPickle as pickle
from train import train_models_generator as T
# from postprocessing import TestSetAnalysis as A

#==============================================================================

# Parameters:
#------------

train_dir='/Users/irshad/work/data/catsdogs/train/'
valid_dir='/Users/irshad/work/data/catsdogs/validation/'
save_dir='../output/'
img_width=150
img_height=150
batch_size=32
nchannels=3
rescale=1./255
shear_range=0.2
zoom_range=0.2
horizontal_flip=True

nlayers_conv=3
filters_conv=[32, 64, 128]
nlayers_dense=2
filters_dense=64
conv_kernel=(3,3)
pooling_kernel=(2,2)
activation='relu'
loss='binary_crossentropy'
metrics=['accuracy']
optimizer='adadelta'
nb_epoch=20
verbose=1
save=True
savefilename='convlayers%i_denselayers%i_nepoch%i.hdf5'\
				%(nlayers_conv, nlayers_dense, nb_epoch)
print_metadata=True

#==============================================================================

ob = T(train_dir, valid_dir, save_dir,\
        img_width, img_height, batch_size, \
        nchannels, rescale, shear_range, \
        zoom_range, horizontal_flip)

hist = ob.train_custom_model(nlayers_conv, filters_conv, \
        nlayers_dense, filters_dense, \
        conv_kernel, pooling_kernel, activation, \
        loss, metrics, optimizer, nb_epoch, verbose, \
        save, savefilename, print_metadata)

#==============================================================================

# obj_train = A(save_dir+savefilename, False)
# obj_train.predict_gen(train_dir, batchsize=batch_size)
# mydict_train = obj_train.get_information_dictionary()
# pickle.dump(mydict_train, \
# 	open(save_dir+savefilename.replace('.hdf5', '_postprocess_train.pkl'), 'w'), -1)

# if valid_dir != None:
# 	obj_valid = A(save_dir+savefilename, False)
# 	obj_valid.predict_gen(valid_dir, batchsize=batch_size)
# 	mydict_valid = obj_valid.get_information_dictionary()
# 	pickle.dump(mydict_valid, \
# 		open(save_dir+savefilename.replace('.hdf5', '_postprocess_valid.pkl'), 'w'), -1)

# pickle.dump(hist.history, open(save_dir+savefilename.replace('.hdf5', '_hist.p'), "w"))

#==============================================================================