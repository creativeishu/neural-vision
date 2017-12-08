import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Dense, Dropout
from keras import backend as K
from keras.optimizers import SGD
from keras.utils import np_utils
from glob import glob

np.random.seed(1234)
from sys import argv, exit

__author__ = 'irshad mohammed'

#==============================================================================

class Finetune(object):
    """
    Class doc string 
    """
    def __init__(self, modelname, train_dir, valid_dir=None, save_dir=None,\
                    img_width=224, img_height=224, batch_size=32, \
                    nchannels=3, rescale=1./255, shear_range=0.2, \
                    zoom_range=0.2, horizontal_flip=True, \
                    loss='binary_crossentropy', optimizer='adadelta', \
                    metrics=['accuracy'], \
                    nb_epoch_top=50, nb_epoch=50, \
                    print_metadata=True, \
                    bf_name='bottleneck_features', \
                    top_name = 'top_model', \
                    final_name = 'finetuned', \
                    pre_name = 'pre_finetuned'):
        """
        Doc string for the constructor
        """
        #----------------------------------------

        self.modelname = modelname
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.nb_epoch_top = nb_epoch_top
        self.nb_epoch = nb_epoch

        #----------------------------------------


        if save_dir==None:
            self.save_dir = train_dir.replace('train', 'savedmodels')
        else:
            self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.bottleneck_train_file = self.save_dir+\
                        '%s_%s_train.npy'%(modelname, bf_name)
        self.bottleneck_valid_file = self.save_dir+\
                        '%s_%s_validation.npy'%(modelname, bf_name)
        self.top_model_file = self.save_dir+'%s_%s.hdf5'%(modelname, top_name)
        self.savefilename = self.save_dir+'%s_%s.hdf5'\
                                %(final_name, modelname)
        self.premodelname = self.save_dir+'%s_%s.hdf5'\
                                %(pre_name, modelname)

        #----------------------------------------

        self.train_generator = self.initialize_generator(train_dir, \
                            img_width, img_height, batch_size, \
                            rescale=rescale, shear_range=shear_range, \
                            zoom_range=zoom_range, \
                            horizontal_flip=horizontal_flip)
        self.nb_class = self.train_generator.num_class
        self.nb_train_samples = self.train_generator.samples
        
        if self.valid_dir != None:
            self.valid_generator = self.initialize_generator(valid_dir, \
                                    img_width, img_height, batch_size, \
                                    rescale=rescale)
            self.nb_valid_samples = self.valid_generator.samples
        else:
            self.valid_generator=None
            self.nb_valid_samples=0

        #----------------------------------------

        self.nTrain, self.nValidation = self.get_samples()

        #----------------------------------------

        if (K.image_data_format()=='channels_last'):
            self.inputshape = (img_width, img_height, nchannels)
        elif (K.image_data_format()=='channels_first'):
            self.inputshape = (nchannels, img_width, img_height)
        else:
            print "Invalid  data format: ", K.image_data_format()
            exit()
        #----------------------------------------

        self.base_model, self.untrainable_layers = \
                        self.load_standard_models_for_training(modelname)

        #----------------------------------------
        if print_metadata:
            self.print_summary()

#------------------------------------------------------------------------------

    def print_summary(self):
        print "============================================"
        print 
        print "Summary"
        print "-------"
        print 
        print "Train data directory: ", self.train_dir
        if self.valid_dir != None:
            print "Vaoidation data directory: : ", self.valid_dir
        print "bottleneck_features will be saved/loaded from directory: ", self.save_dir
        print "Top_model will be saved/loaded as: ", self.top_model_file
        print "pre-Fine tuned model will be saved at: ", self.premodelname
        print "Fine tuned model will be saved at: ", self.savefilename
        print "Using model: ", self.modelname
        print "Data Format: ", K.image_data_format()
        print "Input shape: ", self.inputshape
        print "Number of training samples: ", self.nb_train_samples
        if self.valid_dir != None:
            print "Number of validation samples: ", self.nb_valid_samples
        print "Number of classes: ", self.nb_class
        print "Batch size: ", self.batch_size
        print "Loss: ", self.loss
        print "Optimizer: ", self.optimizer
        print "Metrics: ", self.metrics
        print "Number of epochs for top model: ", self.nb_epoch_top
        print "Number of epochs for fine tuning: ", self.nb_epoch
        print 
        print "============================================"

#------------------------------------------------------------------------------        
        
    def initialize_generator(self, DIR, img_width, img_height, batch_size, \
                                rescale=None, shear_range=0.0, \
                                zoom_range=0.0, horizontal_flip=False, \
                                class_mode='categorical', shuffle=True):
        """
        Doc string for initialize_generator
        """

        datagen = ImageDataGenerator(rescale=rescale, \
                                    shear_range=shear_range, \
                                    zoom_range=zoom_range, \
                                    horizontal_flip=horizontal_flip)
        generator = datagen.flow_from_directory(DIR, \
                                    target_size=(img_width, img_height), \
                                    batch_size=batch_size, \
                                    class_mode=class_mode, \
                                    shuffle=shuffle)
        return generator

#------------------------------------------------------------------------------

    def get_samples(self):
        nTrain = []
        class_folders = glob(self.train_dir+'*')
        for i in range(len(class_folders)):
            files = glob(class_folders[i]+'/*')
            nTrain.append(len(files))

        if self.valid_dir != None:
            nValidation = []
            class_folders = glob(self.valid_dir+'*')
            for i in range(len(class_folders)):
                files = glob(class_folders[i]+'/*')
                nValidation.append(len(files))
        else:
            nValidation = np.zeros((self.nb_class))
        return nTrain, nValidation

#------------------------------------------------------------------------------

    def load_standard_models_for_training(self, model='vgg16', \
                                        weights='imagenet'):
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
                               input_shape=self.inputshape)
            untrainable_layers = 17
        elif model=='vgg16':
            from keras.applications.vgg16 import VGG16
            base_model = VGG16(weights=weights, include_top = False, \
                               input_shape=self.inputshape)
            untrainable_layers = 15
        else:
            print "Valid models are:"
            # print "vgg19, vgg16, inceptionv3, resnet50, xception"
            print "vgg19, vgg16"
            print "Note: xception model is only available in tf backend"
            exit()

        return base_model, untrainable_layers

#------------------------------------------------------------------------------

    def save_bottlebeck_features(self):
        print "============================================"
        print 
        print "Training bottleneck_features for training set"
        print 

        generator = self.initialize_generator(self.train_dir, \
                                self.inputshape[0], self.inputshape[1], \
                                self.batch_size, \
                                rescale=None, shear_range=0.0, \
                                zoom_range=0.0, horizontal_flip=False, \
                                class_mode=None, shuffle=False)
        bottleneck_features_train = self.base_model.predict_generator(\
                                generator,\
                                self.nb_train_samples/self.batch_size+1)
        np.save(open(self.bottleneck_train_file, 'w'), \
                                bottleneck_features_train)

        if self.valid_dir != None:
            print "Training bottleneck_features for validation set"
            generator = self.initialize_generator(self.valid_dir, \
                                    self.inputshape[0], self.inputshape[1], \
                                    self.batch_size, \
                                    rescale=None, shear_range=0.0, \
                                    zoom_range=0.0, horizontal_flip=False, \
                                    class_mode=None, shuffle=False)
            bottleneck_features_valid = self.base_model.predict_generator(\
                                    generator,\
                                    self.nb_valid_samples/self.batch_size+1)
            np.save(open(self.bottleneck_valid_file, 'w'), \
                                    bottleneck_features_valid)
        print 
        print "============================================"

#------------------------------------------------------------------------------

    def one_hot_encode_object_array(self, arr):
        '''One hot encode a numpy array of objects (e.g. strings)'''
        uniques, ids = np.unique(arr, return_inverse=True)
        return np_utils.to_categorical(ids, len(uniques))

#------------------------------------------------------------------------------

    def train_top_model(self, verbose=1):
        print "============================================"    
        print
        print "Training top_model..."
        print 
        train_data = np.load(open(self.bottleneck_train_file))
        train_labels = []
        for i in range(self.nb_class):
            train_labels +=  list([i] * self.nTrain[i])
        train_labels = np.array(train_labels)
        train_labels = self.one_hot_encode_object_array(train_labels)

        if self.valid_dir != None:
            validation_data = np.load(open(self.bottleneck_valid_file))
            validation_labels = []
            for i in range(self.nb_class):
                validation_labels +=  list([i] * self.nValidation[i])
            validation_labels = np.array(validation_labels)
            validation_labels = self.one_hot_encode_object_array(validation_labels)

#        print "Hola: ", train_data.shape, validation_data.shape
        top_model = Sequential()
        top_model.add(Flatten(input_shape=train_data.shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dense(128, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(self.nb_class, activation='sigmoid'))

        top_model.compile(loss=self.loss, \
                            optimizer=self.optimizer, \
                            metrics=self.metrics)  

        if self.valid_dir != None:
            top_model.fit(train_data, train_labels,
                  epochs=self.nb_epoch_top,
                  batch_size=self.batch_size,
                  validation_data=(validation_data, validation_labels), \
                  verbose=verbose)
        else:
            top_model.fit(train_data, train_labels,
                  epochs=self.nb_epoch_top,
                  batch_size=self.batch_size, verbose=verbose)

        top_model.save(self.top_model_file)
        print 
        print "============================================"        
        return top_model

#------------------------------------------------------------------------------

    def fine_tune(self, Train_bottleneck=False, Train_topmodel=False, \
                        verbose=1, savemodel=True, savepremodel=True):

        if Train_bottleneck:
            self.save_bottlebeck_features()

        if Train_topmodel:    
            self.top_model = self.train_top_model(verbose=verbose)
        else:
            print "Loading top model"
            self.top_model = load_model(self.top_model_file)

        model = Model(inputs=self.base_model.input, \
                        outputs=self.top_model(self.base_model.output))           

        for layer in model.layers[:self.untrainable_layers]:
            layer.trainable = False
        model.compile(loss=self.loss, \
                            optimizer=self.optimizer, \
                            metrics=self.metrics)         
        if savepremodel:
            print "Saving pre-model"
            model.save(self.premodelname, overwrite=True)

        print "Now fine tuning... "
        if self.valid_dir != None:
            hist = model.fit_generator(self.train_generator, \
                validation_data=self.valid_generator,\
                steps_per_epoch=self.nb_train_samples/self.batch_size+1, \
                epochs=self.nb_epoch, \
                validation_steps=self.nb_valid_samples/self.batch_size+1, \
                verbose=verbose)
        else:
            hist = model.fit_generator(self.train_generator, \
                steps_per_epoch=self.nb_train_samples/self.batch_size+1, \
                epochs=self.nb_epoch, verbose=verbose)
        if savemodel:
            model.save(self.savefilename, overwrite=True)
            
        return hist

#==============================================================================

if __name__ == "__main__":
    from sys import argv 
    obj = Finetune('vgg16', argv[1], argv[2])
    obj.fine_tune(Train_bottleneck=False, Train_topmodel=False)

