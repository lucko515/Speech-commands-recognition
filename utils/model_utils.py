from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda, BatchNormalization, Conv1D, GRU, TimeDistributed, Activation, Dense, Flatten
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint   
from keras.losses import categorical_crossentropy
import os


def cnn_output_length(input_length, 
                      filter_size, 
                      padding, 
                      stride,
                      dilation=1):
    
    '''
    Calculates output length based on the input sample. NOTE: Used only for architectures with Conv1D layers.

    :param:
        input_length -  Integer, length of the input data  (Example: input.shape[0])
        filter_size - Integer, kernel_size of the Conv layer
        padding - String, Padding version on the Conv layer ("same" or "valid")
        stride - Integer, Conv layer strides size 
        dilation - Integer
    '''
    if input_length is None:
        return None
    assert padding in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding == 'same':
        output_length = input_length
    else:
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride


def ctc_loss(args):
    '''
    More info on CTC: https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c

    Creates CTC (Connectionist Temporal Classification) loss for a speech_to_text model approach.

    :params:
        args - List of params: predictions, labels, input_len and labels_len

    :returns:
        calculated CTC loss based on args.
    '''
    predictions, labels, input_len, labels_len = args
    return K.ctc_batch_cost(labels, predictions, input_len, labels_len)

def categorical_loss(args):
    '''
    Creates Categorical crossentropy loss for a classificaiton based model.

    :params:
        args - List of params: predictions, labels

    :returns:
        calculated categorical_crossentropy loss based on args.
    '''

    predictions, labels = args
    return categorical_crossentropy(labels, predictions)

def add_ctc_loss(model):
    '''
    Adds CTC loss to an model.

    :params:
        model - Keras Model object

    :returns:
        model - Keras Model object with ctc loss added
    '''
    #Creates placeholder/Input layer for labels
    labels = Input(name='labels', shape=(None,), dtype='float32')
    #Creates placeholder/Input layer for lenghts of input features (time steps)
    input_lens = Input(name='input_length', shape=(1,), dtype='int64')
    #Creates placeholder/Input layer for lenghts of labels/targets (in our case number of characters in a target word)
    labels_lens = Input(name='label_length', shape=(1,), dtype='int64')
    
    #Create lambda funciton around model outputs based on labels lenghts
    outputs = Lambda(model.output_length)(input_lens)
    
    #Add CTC Loss to the input model
    loss = Lambda(ctc_loss, output_shape=(1,), name='ctc')([model.output, labels, outputs, labels_lens])
    
    #Create new model instance with all new placeholders/input layers and loss as the output
    model = Model(inputs=[model.input, labels, input_lens, labels_lens], 
                    outputs=loss)
    
    return model

def add_categorical_loss(model, number_of_classes):
    '''
    Adds categorical_crossentropy loss to an model.

    :params:
        model - Keras Model object
        number_of_classes - Integer, number of classes in a dataset (number of words in this case)

    :returns:
        model - Keras Model object with categorical_crossentropy loss added
    '''
    
    #Creates placeholder/Input layer for labels in one_hot_encoded form
    labels = Input(name='labels', shape=(number_of_classes,), dtype='float32')
    
    #Add categorical_crossentropy Loss to the input model
    loss = Lambda(categorical_loss, output_shape=(1,), name='categorical_crossentropy')([model.output, labels])
    
    #Create new model instance with all new placeholders/input layers and loss as the output
    model = Model(inputs=[model.input, labels], outputs=loss)
    
    return model