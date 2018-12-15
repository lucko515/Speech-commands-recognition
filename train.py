
from generator import AudioGenerator
from models import classification_model, speech_to_text_model
from utils.model_utils import *
from keras.optimizers import RMSprop, SGD
import pickle


def train_model(learning_mode,
                model_name,
                save_path,
                pickle_name,
                batch_size,
                spectrogram,
                verbose=1,
                epochs=20,
                optimizer=SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5),
                generate_noisy_data=True,
                number_of_noisy_samples=3,
                audio_padding=True,
                mfcc_dim=13,
                mfcc_features=[0]):
    '''
    Main training function. Call this function to train a model.

    :params:
        learning_mode - String, What type of learning approach you want to take: Speech to text or classification
        model_name - String, Name of the model, used to save models weights
        save_path - String, Name of the folder where models data will be saved
        pickle_name - String, Name of the pickle file used for saving models loss histroy data
        batch_size - Integer
        spectrogram - Boolean, if True data will be generated with spectogram features, otherwise MFCC features will be used
        verbose - Integer, if 1 Keras will show the training process
        epochs - Integer, number of epochs per training session
        optimizer - Keras optimizer object
        generate_noisy_data - Boolean, if True generator will generate noisy data as additional training data
                                       NOTE: This will increase your batch size, if you have 4GB or less RAM don't use this parameter or set batch_size to 32 or less
        number_of_noisy_samples - Integer, number of noisy samples generater PER sample in a batch
        audio_padding -  Boolean, if set to True each sample will be padded with zeros to match 1sec length
        mfcc_dim - Integer, number of MFCC features
        mfcc_features - Integer list, what MFCC features to use Example: [0] will only be using regular MFCC features, [0, 1] -> regular + delta features
    '''

    #Defines data generatorf for the training process
    generator = AudioGenerator(learning_mode=learning_mode,
                               spectrogram=spectrogram,
                               batch_size=batch_size, 
                               mfcc_features=mfcc_features, 
                               mfcc_dim=mfcc_dim,
                               padd_to_sr=audio_padding,
                               generate_noisy_data=generate_noisy_data,
                               number_of_noisy_samples=number_of_noisy_samples)
    
    
    #calculate steps per epoch
    num_train_examples=len(generator.training_files)
    steps_per_epoch = num_train_examples//batch_size

    num_valid_samples = len(generator.validation_paths) 
    validation_steps = num_valid_samples//batch_size
    
    #pretty much hard coded version of features decision
    #TODO: Make this better, because for now it won't handle delta and delta-delta MFCC features
    if spectrogram:
    	features = 161
    else:
    	features = 13

    if learning_mode == 'speech_to_text':
        model = speech_to_text_model(input_dim=features, 
                                     filters=200, 
                                     kernel_size=11, 
                                     strides=2, 
                                     padding='valid',
                                     rnn_units=200, 
                                     output_dim=29) #number of characters

        #Adds CTC loss for the speech_to_text model
        model = add_ctc_loss(model)
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
    
    elif learning_mode == 'classification':
        #if learning_mode is set to classifcation, define the classifcation model with categorical crossentropy loss
        model = classification_model(input_dim=(99, features), 
                                     filters=256, 
                                     kernel_size=1, 
                                     strides=1, 
                                     padding='valid', 
                                     output_dim=len(generator.classes)) #number of classes

        #Adds categorical crossentropy loss for the classification model
        model = add_categorical_loss(model , len(generator.classes))
        #compile the model with choosen loss and optimizer
        model.compile(loss={'categorical_crossentropy': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
        
    #Creates save folder if it doesn't exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    #Defines checkpointer that is responsible for saving the model after N steps of the training
    checkpointer = ModelCheckpoint(filepath=save_path+model_name, verbose=0)

    #Train the choosen model with the data generator
    hist = model.fit_generator(generator=generator.next_train(),            #Calls generators next_train function which generates new batch of training data
                                steps_per_epoch=steps_per_epoch,            #Defines how many training steps are there 
                                epochs=epochs,                              #Defines how many epochs does a training process takes
                                validation_data=generator.next_valid(),     #Calls generators next_valid function which generates new batch of validation data
                                validation_steps=validation_steps,          #Defines how many validation steps are theere
                                callbacks=[checkpointer],                   #Defines all callbacks (In this case we only have molde checkpointer that saves the model)
                                verbose=verbose)                            #If verbose is 1 we can see the training process 
    
    #Save models training history
    with open(save_path + pickle_name, 'wb') as f:
        pickle.dump(hist.history, f)


if __name__ == '__main__':

	print("Training classification model with spectrogram features.")
	train_model(optimizer=RMSprop(),
			learning_mode='classification',
            save_path="saves/",
            model_name="model_clf_only_commands_spectrogram.h5",
            pickle_path='model_clf_only_commands_spectrogram.pickle',
            batch_size=128,
            spectrogram=True,
            number_of_noisy_samples=2, 
            epochs=3)

    '''
	print("Training speech_to_text model with spectrogram features.")
	train_model(optimizer=RMSprop(),
		    learning_mode='speech_to_text',
            save_path="model_speech_hybrid.h5",
            pickle_path='model_speech_hybrid.pickle',
            batch_size=128,
            spectrogram=True,
            number_of_noisy_samples=2, 
            epochs=15)
    '''