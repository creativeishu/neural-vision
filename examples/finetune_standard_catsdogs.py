"""
DESCRIPTION
"""
import sys
sys.path.append('/Users/irshad/github/neural-vision/src/')

import pickle
from finetuning import Finetune as F

#==============================================================================

# Parameters:
#------------

modelname = 'vgg16'
train_dir = '/Users/irshad/work/data/catsdogs/train/'
valid_dir='/Users/irshad/work/data/catsdogs/validation/'
save_dir='../output/'
img_width=224
img_height=224
batch_size=32
nchannels=3
rescale=1./255
shear_range=0.2
zoom_range=0.2
horizontal_flip=True
loss='binary_crossentropy'
optimizer='adadelta'
metrics=['accuracy']
nb_epoch_top=50
nb_epoch=50
print_metadata=True
bf_name='bottleneck_features'
top_name = 'top_model'
final_name = 'finetuned'
pre_name = 'pre_finetuned'

Train_bottleneck=True
Train_topmodel=True
verbose=1
savemodel=True
savepremodel=True

#==============================================================================

ob = F(modelname, train_dir, valid_dir, save_dir,\
        img_width, img_height, batch_size, \
        nchannels, rescale, shear_range, \
        zoom_range, horizontal_flip, \
        loss, optimizer, metrics, \
        nb_epoch_top, nb_epoch, \
        print_metadata, 
        bf_name, top_name, final_name, pre_name)

hist = ob.fine_tune(Train_bottleneck, Train_topmodel, \
                    verbose, savemodel, savepremodel)

pickle.dump(hist.history, open(savefilename.replace('.hdf5', '_hist.p'), "w"))

#==============================================================================