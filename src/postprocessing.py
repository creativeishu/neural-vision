"""
a long description
"""

import numpy as np 
import os
import cv2
from sys import exit, argv
import matplotlib.pyplot as plt 

from keras.models import load_model
from keras.models import Model


import numpy as np 
import matplotlib.pyplot as plt 

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
import cv2
from glob import glob

__author__ = "Irshad Mohammed"

#==============================================================================

class image_analysis(object):
	"""
	class string
	"""

	def __init__(self, model='vgg19'):
		"""
		doc string constructor
		"""
		firstlayer_index = 0
		if model=='vgg19':
		    from keras.applications.vgg19 import VGG19
		    self.model = VGG19(weights='imagenet', include_top = True)
		elif model=='vgg16':
		    from keras.applications.vgg16 import VGG16
		    self.model = VGG16(weights='imagenet', include_top = True)
		elif model=='inceptionv3':
			from keras.applications.inception_v3 import InceptionV3
			self.model = InceptionV3(weights='imagenet', include_top = True)
		elif model=='resnet50':
		    from keras.applications.resnet50 import ResNet50
		    self.model = ResNet50(weights='imagenet', include_top = True)
		elif model=='xception':
		    from keras.applications.xception import Xception
		    self.model = Xception(weights='imagenet', include_top = True)        
		elif model.endswith('.hdf5'):
			self.model = load_model(model)
			firstlayer_index = 1
		else:
		    print "Valid models are:"
		    print "vgg19, vgg16, inceptionv3, resnet50, xception"
		    print "xception/inceptionv3 model is only available in tf backend"
		    print "Or provide path to a saved model in .hdf5 format"
		    exit()
		self.inputshape = self.model.layers[firstlayer_index].output_shape[1:]

#------------------------------------------------------------------------------

	def get_labels(self, path):
		"""
		my doc string
		"""
		labels = []
		f = open(path, 'r')
		while True:
			line = f.readline()
			if not line:
				break
			labels.append(line)
		f.close()
		return labels

#------------------------------------------------------------------------------

	def get_image_input(self, imagepath):
		"""
		my doc string
		"""		
		shape = self.inputshape[:2]
		im = cv2.resize(cv2.imread(imagepath), shape).astype(np.float32)
		im[:,:,0] -= 103.939
		im[:,:,1] -= 116.779
		im[:,:,2] -= 123.68
		im = np.expand_dims(im, axis=0)
		return im

#------------------------------------------------------------------------------

	def get_image_category(self, imagepath=None, labelpath=None, k=5):
		if imagepath==None or labelpath==None:
			print "Usage: predict_image(imagepage, labelpath)"
			exit()
		im = self.get_image_input(imagepath)
		out = self.model.predict(im)
		idxs = np.argsort(out[0])[::-1][:k]
		labels = self.get_labels(labelpath)
		results = np.array(labels, dtype='str')[idxs]
		probs = np.array(out[0,idxs], dtype='float')
		dictionary = {}
		for i in range(k):
			dictionary[results[i].replace('\n','')] = probs[i]
		return dictionary

#------------------------------------------------------------------------------

	def get_features(self, img_path, layername='fc1'):
		model = Model(inputs=self.model.input, \
				outputs=self.model.get_layer(layername).output)
		im = self.get_image_input(img_path)
		features = model.predict(im)
		return features

#------------------------------------------------------------------------------

	def get_features_vector(self, img_path, layername='fc1'):
		features = self.get_features(img_path, layername)
		return np.ndarray.flatten(features)

#------------------------------------------------------------------------------

	def get_layernames(self):
	    allnames = []
	    for i in range(len(self.model.layers)):
	        names = self.model.layers[i].name
	        allnames.append(names)
	    return allnames		

#------------------------------------------------------------------------------

	def plot_conv_features(self, image, layer=None, \
						save=False, savefilename='features.eps'):
		if layer==None:
			names = self.get_layernames()
			for i in range(len(names)):
				if 'conv' in names[i]:
					layer=names[i]
					break
		elif type(layer)==int:
			names = self.get_layernames()
			layer = names[layer]

		if not (('conv' in layer) or ('pool' in layer)):
			print "please provide name or index of convolution/pooling layer"
			exit()

		features = self.get_features(image, layer)
		N = features.shape[-1]
		nrow = int(N**0.5)
		ncol = int(N**0.5)

		print "================================================="
		print "Layer name: ", layer
		print "Features shape: ", features.shape
		print "Number of rows and columns: ", nrow, ncol
		print "================================================="

		f, axarr = plt.subplots(nrow, ncol, \
	    	sharex=True, sharey=True, figsize=(20,20))
		# f.suptitle('$\mathtt{%s}$'%layer.replace('_', " "), fontsize=22)
		f.subplots_adjust(wspace=0.02, hspace=0.02)
		for i in range(nrow):
		    for j in range(ncol):
		        axarr[i,j].imshow(features[0,:,:,i*ncol+j], cmap='jet')
		        axarr[i,j].set_xticks([], [])
		        axarr[i,j].set_yticks([], [])
		if save:
			f.savefig(savefilename)
		else:
			plt.show()

#------------------------------------------------------------------------------

	def plot_all_conv_features(self, image, DIR='figures/'):
		if not os.path.exists(DIR):
			os.mkdir(DIR)
		names = self.get_layernames()
		for i in range(len(names)):
		    if ('conv' in names[i]) or ('pool' in names[i]):
		    	filename = DIR+names[i]+'.eps'
		        self.plot_conv_features(image, names[i], \
		        	save=True, savefilename=filename)

#==============================================================================
#==============================================================================
#==============================================================================

class TestSetAnalysis(object):
	"""
	class string
	"""

	def __init__(self, model='vgg19', show=True):
		"""
		doc string constructor
		"""
		firstlayer_index = 0
		if model=='vgg19':
		    from keras.applications.vgg19 import VGG19
		    self.model = VGG19(weights='imagenet', include_top = True)
		elif model=='vgg16':
		    from keras.applications.vgg16 import VGG16
		    self.model = VGG16(weights='imagenet', include_top = True)
		elif model=='inceptionv3':
			from keras.applications.inception_v3 import InceptionV3
			self.model = InceptionV3(weights='imagenet', include_top = True)
		elif model=='resnet50':
		    from keras.applications.resnet50 import ResNet50
		    self.model = ResNet50(weights='imagenet', include_top = True)
		elif model=='xception':
		    from keras.applications.xception import Xception
		    self.model = Xception(weights='imagenet', include_top = True)        
		elif model.endswith('.hdf5'):
			self.model = load_model(model)
			firstlayer_index = 1
		else:
		    print "Valid models are:"
		    print "vgg19, vgg16, inceptionv3, resnet50, xception"
		    print "xception/inceptionv3 model is only available in tf backend"
		    print "Or provide path to a saved model in .hdf5 format"
		    exit()
		if show:
			print self.model.summary()
		self.inputshape = self.model.layers[firstlayer_index].output_shape[1:]

#------------------------------------------------------------------------------

	def predict_gen(self, data_dir, batchsize=32, rescale=1.0/255):
		self.data_dir = data_dir
		datagen = ImageDataGenerator(rescale=rescale)
		self.generator = datagen.flow_from_directory(self.data_dir, \
								target_size=self.inputshape[:2], \
		                        batch_size=batchsize, \
		                        class_mode='categorical', \
		                        shuffle=False)

		nfiles = []
		class_folders = glob(self.data_dir+'*')
		for i in range(len(class_folders)):
		    files = glob(class_folders[i]+'/*')
		    nfiles.append(len(files))

		samples = self.generator.samples
		self.nb_class = self.generator.num_class
		self.predictions = self.model.predict_generator(self.generator, \
												samples/batchsize+1)
		self.predictions = self.predictions[:samples, :]
		self.predict_labels = np.argmax(self.predictions, axis=1)
		self.true_labels = []
		for i in range(self.nb_class):
			self.true_labels +=  list([i] * nfiles[i])

		self.confusion_matrix = confusion_matrix(\
										self.true_labels, \
										self.predict_labels)

		if self.nb_class==2:
			self.FPR, self.TPR, thresholds = roc_curve(\
										self.true_labels, \
										self.predictions[:,1])
			self.roc_auc = roc_auc_score(\
										self.true_labels, \
										self.predictions[:,1])
			self.get_cm_index()

#------------------------------------------------------------------------------

	def predict_array(self, xdata, ydata, batchsize=32, rescale=1.0/255):

		# samples = self.generator.samples
		# self.nb_class = self.generator.num_class
		self.predictions = self.model.predict(xdata*rescale, batch_size=batchsize)
		# self.predictions = self.predictions[:samples, :]
		self.predict_labels = np.argmax(self.predictions, axis=1)
		self.true_labels = np.argmax(ydata, axis=1)

		self.confusion_matrix = confusion_matrix(\
										self.true_labels, \
										self.predict_labels)

		self.FPR, self.TPR, thresholds = roc_curve(\
									self.true_labels, \
									self.predictions[:,1])
		self.roc_auc = roc_auc_score(\
									self.true_labels, \
									self.predictions[:,1])
		self.get_cm_index()			

#------------------------------------------------------------------------------

	def get_information_dictionary(self):
		mydict = {
			"FPR": self.FPR,
			"TPR": self.TPR,
			"predictions": self.predictions,
			"true_labels": self.true_labels,
			"predict_labels": self.predict_labels,
			"roc_auc": self.roc_auc,
			"confusion_matrix": self.confusion_matrix
		}
		return mydict

#------------------------------------------------------------------------------

	def plot_confusion_matrix(self, cmap='Blues', \
								save=False, savename='cm.png'):
		plt.figure(figsize=(8,8))
		matrix = np.zeros(self.confusion_matrix.shape)
		for i in range(len(matrix)):
			matrix[i] = self.confusion_matrix[i]/\
						float(np.sum(self.confusion_matrix[i]))
		plt.imshow(matrix, cmap=cmap)
		plt.xticks([], [])
		plt.yticks([], [])
		plt.clim(0, 1)
		if save:
			print "Now saving confusion matrix figure"
			plt.savefig(savename)
		else:
			plt.show()
		return matrix

#------------------------------------------------------------------------------

	def plot_roc_curve(self, \
					save=False, savename='roc.png'):
		plt.figure(figsize=(8,8))
		plt.plot(self.FPR, self.TPR, 'k', lw=2)
		plt.plot(self.FPR, self.FPR, 'k', lw=0.5)
		plt.axhline(y=1, color='k', ls=':', lw=0.5)
		plt.axvline(x=0, color='k', ls=':', lw=0.5)
		plt.xlim(-0.01,1)
		plt.ylim(0,1.01)
		plt.xlabel('$\mathtt{FalsePositiveRate}$', fontsize=22)
		plt.ylabel('$\mathtt{TruePositiveRate}$', fontsize=22)
		if save:
			f.savefig(savefigname)
		else:
			plt.show()

#------------------------------------------------------------------------------

	def plot_samples(self, ind_arr, title, N=100, ncol=15, \
						save=False, savefigname='samples.eps'):

	    ind_arr = np.random.choice(ind_arr, size=N, replace=False)
	    names = np.array(self.generator.filenames)[ind_arr]
	    N = N - N%ncol
	    print N
	    nrow = N/ncol
	    f, axarr = plt.subplots(nrow, ncol, sharex=True, sharey=True, \
	    						figsize=(ncol, nrow))
	    f.subplots_adjust(wspace=0.0, hspace=0)
	    f.suptitle("$\mathtt{%s}$"%title, fontsize=22)

	    for i in range(nrow):
	        for j in range(ncol):
	            axarr[i,j].imshow(cv2.imread(self.data_dir+names[i*ncol+j]))
	            axarr[i,j].set_xticks([], [])
	            axarr[i,j].set_yticks([], [])
	    if save:
	    	f.savefig(savefigname)
	    else:
	    	plt.show()

#------------------------------------------------------------------------------

	def get_cm_index(self):
	    self.tp = []
	    self.tn = []
	    self.fp = []
	    self.fn = []
	    for i in range(len(self.true_labels)):
	        if self.true_labels[i]==1 and self.predict_labels[i]==1:
	            self.tp.append(i)
	        elif self.true_labels[i]==0 and self.predict_labels[i]==0:
	            self.tn.append(i)
	        elif self.true_labels[i]==0 and self.predict_labels[i]==1:
	            self.fp.append(i)
	        elif self.true_labels[i]==1 and self.predict_labels[i]==0:
	            self.fn.append(i)

#------------------------------------------------------------------------------

	def plot_all(self):
		self.plot_confusion_matrix()
		if self.nb_class==2:
			self.plot_roc_curve()
			self.plot_samples(self.tp, 'TruePositive')
			self.plot_samples(self.fp, 'FalsePositive')
			self.plot_samples(self.tn, 'TrueNegative')
			self.plot_samples(self.fn, 'FalseNegative')

#==============================================================================

if __name__ == "__main__":
	if len(argv)==3:
		from sys import argv
		model = argv[1]
		data_dir = argv[2]
		obj = TestSetAnalysis(model, False)
		obj.predict_gen(data_dir, batchsize=32)
		obj.plot_all()
	else:
		print "Usage: Either import this class, or do:"
		print "python imagepostprocessing <model_path> <data_directory>"

