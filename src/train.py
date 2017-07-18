"""
A long description of how to use this file

To check the associated methods, just do : dir(object_name)
To read the doc string of a function, do: print my_function.__doc__

export PYTHONPATH=<PATH_TO_THIS_FOLDER>:$PYTHONPATH
"""
import os 
import numpy as np 

from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.layers import Flatten, Dense, Dropout
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, Sequential

from sys import exit

__author__ = "Irshad Mohammed"

#==============================================================================

class train_models_array(object):
    """
    My main docstring
    """
    def __init__(self, xdata, ydata, save_dir=None, rescale=1.0/255.0):
        """
        Doc string for the constructor
        """
        self.xdata = xdata*rescale
        self.ydata = ydata
        print self.xdata.shape
        print self.ydata.shape

        self.nb_train_samples = self.xdata.shape[0]
        img_width = self.xdata.shape[1]
        img_height = self.xdata.shape[2]
        nchannels = self.xdata.shape[3]
        self.nb_class = self.ydata.shape[1]


        if save_dir==None:
            self.save_dir = xname.replace('.npy', '/savedmodels')
        else:
            self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)


        if (K.image_data_format()=='channels_last'):
            self.inputshape = (img_width, img_height, nchannels)
        elif (K.image_data_format()=='channels_first'):
            self.inputshape = (nchannels, img_width, img_height)
        else:
            print "Invalid  data format: ", K.image_data_format()
            exit()

#------------------------------------------------------------------------------

    def print_summary(self):
        print
        print "================================================================"
        print 
        print "Training data shape: ", self.xdata.shape
        print "Output will be saved at: ", self.save_dir
        print "Input shape: ", self.inputshape
        print "Number of training samples: ", self.nb_train_samples

#------------------------------------------------------------------------------

    def load_custom_models_for_training(self, inputshape=(224,224,3), \
                        nlayers_conv=3, filters_conv=64, \
                        nlayers_dense=2, filters_dense=256, \
                        conv_kernel=(3,3), pooling_kernel=(2,2), \
                        activation='relu'):
        """
        My Doc string 
        """

        if type(filters_conv)==int:
            filters_conv=[filters_conv]*nlayers_conv
        elif not(type(filters_conv)==list and \
                    (nlayers_conv==len(filters_conv))):
            print "filters_conv can only be an integer "
            print "or list of length nlayers_conv"
            exit()

        if type(filters_dense)==int:
            filters_dense=[filters_dense]*nlayers_dense
        elif not(type(filters_dense)==list and \
                    (nlayers_dense==len(filters_dense))):
            print "filters_dense can only be an integer "
            print "or list of length nlayers_dense"
            exit()

        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=inputshape))
        for i in range(nlayers_conv):
            model.add(Conv2D(filters_conv[i], conv_kernel, 
                activation=activation, name='conv2d_%i'%(i+1)))
            model.add(MaxPooling2D(pooling_kernel, strides=(2, 2)))
            if i<nlayers_conv-1:
                model.add(ZeroPadding2D((1,1)))
        model.add(Flatten())

        for i in range(nlayers_dense):
            model.add(Dense(filters_dense[i], activation=activation))
            model.add(Dropout(0.5))

        model.add(Dense(self.nb_class, activation='sigmoid'))
        return model 


#------------------------------------------------------------------------------        

    def train_custom_model(self, \
                    nlayers_conv=5, filters_conv=32, \
                    nlayers_dense=2, filters_dense=32, \
                    conv_kernel=(3,3), pooling_kernel=(2,2), \
                    val_split=0.2, batch_size=32,\
                    activation='relu', \
                    loss='binary_crossentropy', \
                    metrics=['accuracy'], \
                    optimizer='adadelta', \
                    nb_epoch=50, verbose=1, \
                    save=False, savefilename='mymodel.hdf5', \
                    print_metadata=True):
        """
        Arguements:
        -----------
        nlayers_conv=3, filters_conv=64, \
        nlayers_dense=2, filters_dense=256, \
        conv_kernel=(3,3), pooling_kernel=(2,2), \
        activation='relu', \
        loss='categorical_crossentropy', \
        metrics=['accuracy'], \
        optimizer='adadelta', \
        nb_epoch=50, verbose=1, \
        save=False, savefilename='mymodel.hdf5', \
        print_metadata=True
        """

        mymodel = self.load_custom_models_for_training(\
                    inputshape=self.inputshape, \
                    nlayers_conv=nlayers_conv, filters_conv=filters_conv, \
                    nlayers_dense=nlayers_dense, filters_dense=filters_dense, \
                    activation=activation)
        mymodel.compile(loss=loss, metrics=metrics, optimizer=optimizer)

        if print_metadata:
            self.print_summary()            
            print "Number of convolutional layers: ", nlayers_conv
            print "Number of convolutional filters: ", filters_conv
            print "Number of dense layers: ", nlayers_dense
            print "Number of dense filters: ", filters_dense
            print "Activation functions: ", activation
            print "Loss: ", loss
            print "Metrics: ", metrics
            print "Optimizer: ", optimizer
            print "Number of epochs: ", nb_epoch
            print "Model file: ", self.save_dir+savefilename
            print mymodel.summary()

        hist = mymodel.fit(self.xdata, self.ydata, \
                    validation_split=val_split, \
                    batch_size=batch_size, epochs=nb_epoch, \
                    verbose=verbose)
        if save:
            mymodel.save(self.save_dir+savefilename)
        return hist

#==============================================================================


class train_models_generator(object):
    """
    My main docstring
    """
    def __init__(self, train_dir, valid_dir=None, save_dir=None,\
                    img_width=224, img_height=224, batch_size=32, \
                    nchannels=3, rescale=1./255, shear_range=0.2, \
                    zoom_range=0.2, horizontal_flip=True):
        """
        Doc string for the constructor
        """
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        if save_dir==None:
            self.save_dir = train_dir.replace('train', 'savedmodels')
        else:
            self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.batch_size = batch_size
        self.train_generator = self.initialize_generator(train_dir, \
                            img_width, img_height, batch_size, \
                            rescale=rescale, shear_range=shear_range, \
                            zoom_range=zoom_range, \
                            horizontal_flip=horizontal_flip)
        self.nb_class = self.train_generator.num_class
        self.nb_train_samples = self.train_generator.samples
        
        if valid_dir != None:
            self.valid_generator = self.initialize_generator(valid_dir, \
                                    img_width, img_height, batch_size, \
                                    rescale=rescale)
            self.nb_valid_samples = self.valid_generator.samples
        else:
            self.valid_generator=None
            self.nb_valid_samples=0

        if (K.image_data_format()=='channels_last'):
            self.inputshape = (img_width, img_height, nchannels)
        elif (K.image_data_format()=='channels_first'):
            self.inputshape = (nchannels, img_width, img_height)
        else:
            print "Invalid  data format: ", K.image_data_format()
            exit()

#------------------------------------------------------------------------------

    def print_summary(self):
        print
        print "================================================================"
        print 
        print "Training data: ", self.train_dir
        print "Validation data: ", self.valid_dir
        print "Output will be saved at: ", self.save_dir
        print "Input shape: ", self.inputshape
        print "Number of training samples: ", self.nb_train_samples
        print "Number of validation samples: ", self.nb_valid_samples
        print "Batch size: ", self.batch_size

#------------------------------------------------------------------------------        
        
    def initialize_generator(self, dir, img_width, img_height, batch_size, \
                                rescale=None, shear_range=0.0, \
                                zoom_range=0.0, horizontal_flip=False):
        """
        Doc string for initialize_generator
        """

        datagen = ImageDataGenerator(rescale=rescale, \
                                    shear_range=shear_range, \
                                    zoom_range=zoom_range, \
                                    horizontal_flip=horizontal_flip)
        generator = datagen.flow_from_directory(dir, \
                                    target_size=(img_width, img_height), \
                                    batch_size=batch_size, \
                                    class_mode='categorical')
        return generator

#------------------------------------------------------------------------------

    def load_standard_models_for_training(self, model='vgg19', weights=None, \
                                            inputshape=(224,224,3)):
        """
        Arguements:
            model: vgg19, vgg16, inceptionv3, resnet50, xception
            nb_class: Number of classes 
            inputshape: (img_width, img_height, n_channels)
            weights: imagenet or None
            
        Returns: 
            A model, neural network architecture. 
        """

        if model=='vgg19':
            from keras.applications.vgg19 import VGG19
            base_model = VGG19(weights=weights, include_top = False, \
                               input_shape=inputshape)
        elif model=='vgg16':
            from keras.applications.vgg16 import VGG16
            base_model = VGG16(weights=weights, include_top = False, \
                               input_shape=inputshape)
        elif model=='inceptionv3':
            from keras.applications.inception_v3 import InceptionV3
            base_model = InceptionV3(weights=weights, include_top = False, \
                               input_shape=inputshape)
        elif model=='resnet50':
            from keras.applications.resnet50 import ResNet50
            base_model = ResNet50(weights=weights, include_top = False, \
                               input_shape=inputshape)
        elif model=='xception' and K.backend=='tensorflow':
            from keras.applications.xception import Xception
            base_model = Xception(weights=weights, include_top = False, \
                               input_shape=inputshape)        
        else:
            print "Valid models are:"
            print "vgg19, vgg16, inceptionv3, resnet50, xception"
            print "Note: xception model is only available in tf backend"
            exit()

        x = Flatten()(base_model.output)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.nb_class, activation = 'softmax')(x)
        model = Model(inputs = base_model.input, outputs = predictions)
        return model

#------------------------------------------------------------------------------

    def load_custom_models_for_training(self, inputshape=(224,224,3), \
                        nlayers_conv=3, filters_conv=64, \
                        nlayers_dense=2, filters_dense=256, \
                        conv_kernel=(3,3), pooling_kernel=(2,2), \
                        activation='relu'):
        """
        My Doc string 
        """

        if type(filters_conv)==int:
            filters_conv=[filters_conv]*nlayers_conv
        elif not(type(filters_conv)==list and \
                    (nlayers_conv==len(filters_conv))):
            print "filters_conv can only be an integer "
            print "or list of length nlayers_conv"
            exit()

        if type(filters_dense)==int:
            filters_dense=[filters_dense]*nlayers_dense
        elif not(type(filters_dense)==list and \
                    (nlayers_dense==len(filters_dense))):
            print "filters_dense can only be an integer "
            print "or list of length nlayers_dense"
            exit()

        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=inputshape))
        for i in range(nlayers_conv):
            model.add(Conv2D(filters_conv[i], conv_kernel, 
                activation=activation, name='conv2d_%i'%(i+1)))
            model.add(MaxPooling2D(pooling_kernel, strides=(2, 2)))
            if i<nlayers_conv-1:
                model.add(ZeroPadding2D((1,1)))
        model.add(Flatten())

        for i in range(nlayers_dense):
            model.add(Dense(filters_dense[i], activation=activation))
            model.add(Dropout(0.5))

        model.add(Dense(self.nb_class, activation='sigmoid'))
        return model 

#------------------------------------------------------------------------------

    def train_standard_model(self, model='vgg19', weights=None, \
                    loss='binary_crossentropy', \
                    metrics=['accuracy'], \
                    optimizer='adadelta', \
                    nb_epoch=50, verbose=1, \
                    save=False, savefilename='mymodel.hdf5', \
                    print_metadata=True):
        """
        Arguements:
        -----------
        model='vgg19', weights=None, \
        loss='categorical_crossentropy', \
        metrics=['accuracy'], \
        optimizer='adadelta', \
        nb_epoch=50, verbose=1, \
        save=False, savefilename='mymodel.hdf5', \
        print_metadata=True
        """

        mymodel = self.load_standard_models_for_training(model, \
                            weights=weights, \
                            inputshape=self.inputshape)
        mymodel.compile(loss=loss, metrics=metrics, optimizer=optimizer)

        if print_metadata:
            self.print_summary()
            print "Model name: ", model
            print "Loss: ", loss
            print "Metrics: ", metrics
            print "Optimizer: ", optimizer
            print "Number of epochs: ", nb_epoch
            print "Model file: ", self.save_dir+savefilename
            print mymodel.summary()

            hist = mymodel.fit_generator(self.train_generator, \
                    validation_data=self.valid_generator, \
                    steps_per_epoch=self.nb_train_samples/self.batch_size+1, \
                    validation_steps=self.nb_valid_samples/self.batch_size+1,\
                    epochs=nb_epoch, verbose=verbose)
        if save:
            mymodel.save(self.save_dir+savefilename)
        return hist

#------------------------------------------------------------------------------        

    def train_custom_model(self, \
                    nlayers_conv=5, filters_conv=32, \
                    nlayers_dense=2, filters_dense=32, \
                    conv_kernel=(3,3), pooling_kernel=(2,2), \
                    activation='relu', \
                    loss='binary_crossentropy', \
                    metrics=['accuracy'], \
                    optimizer='adadelta', \
                    nb_epoch=50, verbose=1, \
                    save=False, savefilename='mymodel.hdf5', \
                    print_metadata=True):
        """
        Arguements:
        -----------
        nlayers_conv=3, filters_conv=64, \
        nlayers_dense=2, filters_dense=256, \
        conv_kernel=(3,3), pooling_kernel=(2,2), \
        activation='relu', \
        loss='categorical_crossentropy', \
        metrics=['accuracy'], \
        optimizer='adadelta', \
        nb_epoch=50, verbose=1, \
        save=False, savefilename='mymodel.hdf5', \
        print_metadata=True
        """

        mymodel = self.load_custom_models_for_training(\
                    inputshape=self.inputshape, \
                    nlayers_conv=nlayers_conv, filters_conv=filters_conv, \
                    nlayers_dense=nlayers_dense, filters_dense=filters_dense, \
                    activation=activation)
        mymodel.compile(loss=loss, metrics=metrics, optimizer=optimizer)

        if print_metadata:
            self.print_summary()            
            print "Number of convolutional layers: ", nlayers_conv
            print "Number of convolutional filters: ", filters_conv
            print "Number of dense layers: ", nlayers_dense
            print "Number of dense filters: ", filters_dense
            print "Activation functions: ", activation
            print "Loss: ", loss
            print "Metrics: ", metrics
            print "Optimizer: ", optimizer
            print "Number of epochs: ", nb_epoch
            print "Model file: ", self.save_dir+savefilename
            print mymodel.summary()

        hist = mymodel.fit_generator(self.train_generator, \
                    validation_data=self.valid_generator, \
                    steps_per_epoch=self.nb_train_samples/self.batch_size+1, \
                    validation_steps=self.nb_valid_samples/self.batch_size+1,\
                    epochs=nb_epoch, verbose=verbose)
        if save:
            mymodel.save(self.save_dir+savefilename)
        return hist

#==============================================================================

if __name__=="__main__":
    # print train_models_generator.__doc__
    # print train_models_generator.train_custom_model.__doc__
    # print train_models_generator.train_standard_model.__doc__

    #------------------------------------------------------------------------------

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


    xname = '/Users/mohammed/Dropbox/deeplensing/Data/Simulation/SLChallenge/imadjust.npy'
    yname = '/Users/mohammed/Dropbox/deeplensing/Data/Simulation/SLChallenge/classification.npy'

    xdata = np.load(xname)
    ydata = np.load(yname)
    xdata = np.swapaxes(xdata, 1,3)
    xdata = np.swapaxes(xdata, 1,2)
    ydata = to_categorical(ydata, 2)
    save_dir = '/Users/mohammed/Desktop/'
    obj = train_models_array(xdata, ydata, save_dir=save_dir)
    obj.train_custom_model()
