from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda, BatchNormalization, Conv1D, GRU, TimeDistributed, Activation, Dense, Flatten
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint   
from keras.losses import categorical_crossentropy

from utils.model_utils import *
import os

def speech_to_text_model(input_dim, 
                         filters, 
                         kernel_size, 
                         strides,
                         padding, 
                         rnn_units, 
                         output_dim=29,
                         cell=GRU, 
                         activation='relu'):
    """ 
    Creates simple Conv-RNN model used for speech_to_text approach.

    :params:
    	input_dim - Integer, size of inputs (Example: 161 if using spectrogram, 13 for mfcc)
    	filters - Integer, number of filters for the Conv1D layer
		kernel_size - Integer, size of kernel for Conv layer
		strides - Integer, stride size for the Conv layer
		padding - String, padding version for the Conv layer ('valid' or 'same')
		rnn_units - Integer, number of units/neurons for the RNN layer(s)
		output_dim - Integer, number of output neurons/units at the output layer
							  NOTE: For speech_to_text approach, this number will be number of characters that may occur
		cell - Keras function, for a type of RNN layer * Valid solutions: LSTM, GRU, BasicRNN
		activation - String, activation type at the RNN layer

	:returns:
		model - Keras Model object

    """

    #Defines Input layer for the model
    input_data = Input(name='inputs', shape=(None, input_dim))

	#Defines 1D Conv block (Conv layer +  batch norm)
    conv_1d = Conv1D(filters, 
                     kernel_size, 
                     strides=strides, 
                     padding=padding,
                     activation='relu',
                     name='conv1d')(input_data)
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)

    #Defines block (RNN layer + batch norm)
    simp_rnn = cell(rnn_units, 
                   activation=activation,
                   return_sequences=True, 
                   implementation=2, 
                   name='rnn')(bn_cnn)

    bn_rnn = BatchNormalization(name='bn_rnn_1d')(simp_rnn)

    #Apply Dense layer to each time step of the RNN with TimeDistributed function
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)

    #Define model predictions with softmax activation
    y_pred = Activation('softmax', name='softmax')(time_dense)

    #Defines Model itself, and use lambda function to define output length based on inputs
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(x, kernel_size, padding, strides)
    print(model.summary())
    return model


def classification_model(input_dim, 
						 filters, 
						 kernel_size, 
						 strides,
    					 padding, 
    					 rnn_units=256, 
    					 output_dim=30, 
    					 dropout_rate=0.5, 
    					 cell=GRU, 
    					 activation='tanh'):
    """ 
    Creates simple Conv-Bi-RNN model used for word classification approach.

    :params:
    	input_dim - Integer, size of inputs (Example: 161 if using spectrogram, 13 for mfcc)
    	filters - Integer, number of filters for the Conv1D layer
		kernel_size - Integer, size of kernel for Conv layer
		strides - Integer, stride size for the Conv layer
		padding - String, padding version for the Conv layer ('valid' or 'same')
		rnn_units - Integer, number of units/neurons for the RNN layer(s)
		output_dim - Integer, number of output neurons/units at the output layer
							  NOTE: For speech_to_text approach, this number will be number of characters that may occur
		dropout_rate - Float, percentage of dropout regularization at each RNN layer, between 0 and 1
		cell - Keras function, for a type of RNN layer * Valid solutions: LSTM, GRU, BasicRNN
		activation - String, activation type at the RNN layer

	:returns:
		model - Keras Model object

    """

    #Defines Input layer for the model
    input_data = Input(name='inputs', shape=(None, input_dim))

    #Defines 1D Conv block (Conv layer +  batch norm)
    conv_1d = Conv1D(filters, 
    				 kernel_size, 
                     strides=strides, 
                     padding=padding,
                     activation='relu',
                     name='layer_1_conv',
                     dilation_rate=1)(input_data)
    conv_bn = BatchNormalization(name='conv_batch_norm')(conv_1d)

    #Defines Bi-Directional RNN block (Bi-RNN layer + batch norm)
    layer = cell(rnn_units, activation=activation,
                return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate)(conv_bn)
    layer = BatchNormalization(name='bt_rnn_1')(layer)

    #Defines Bi-Directional RNN block (Bi-RNN layer + batch norm)
    layer = cell(rnn_units, activation=activation,
                return_sequences=True, implementation=2, name='final_layer_of_rnn')(layer)
    layer = BatchNormalization(name='bt_rnn_final')(layer)
    
    #Apply Dense layer to each time step of the RNN with TimeDistributed function
    time_dense = TimeDistributed(Dense(output_dim))(layer)

    #Define model predictions with softmax activation
    y_pred = Activation('softmax', name='softmax')(time_dense)

    #Defines Model itself, and use lambda function to define output length based on inputs
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(x, kernel_size, padding, strides)

    print(model.summary())
    return model