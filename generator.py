from utils.audio_utils import compute_mfcc_features, mfcc_pack, generate_spectrogram
from utils.utils import word_to_int, character_mapper
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal

import numpy as np
import librosa
import glob
import os


class AudioGenerator():
    
    def __init__(self,
                 learning_mode='classification',
                 mode='train',
                 batch_size=128,
                 step=10,
                 window_size=20,
                 mfcc_dim=13,
                 mfcc_features=[0],
                 sample_rate=16000,
                 padd_to_sr=True,
                 spectrogram=False,
                 generate_noisy_data=True,
                 number_of_noisy_samples=3,
                 training_dataset_path='data/train/audio/',
                 background_path='data/train/_background_noise_/', 
                 valid_paths='data/train/validation_list.txt',
                 testing_paths='data/train/testing_list.txt'):
        '''
        Audio data generator

        :params:
            learning_mode - String, type of learning approach you want to take: Speech to text or classification
            mode - String, defines models phase (train, test, valid)
            batch_size - Integer
            step - Integer,  number of milliseconds (points) to move the kernel,
                    if the step_size == window_size there is no overlapping beteween signal segments
            window_size - Integer, number of milliseconds that we will consider at the time
            mfcc_dim - Integer, number of MFCC features
            mfcc_features - Integer list, what MFCC features to use Example: [0] will only be using regular MFCC features, [0, 1] -> regular + delta features
            sample_rate - Integer
            padd_to_sr - Boolean, if set to True each sample will be padded with zeros to match sample_rate  
            spectrogram - Boolean, if True data will be generated with spectogram features, otherwise MFCC features will be used
            generate_noisy_data - Boolean, Boolean, if True generator will generate noisy data as additional training data
                                       NOTE: This will increase your batch size, if you have 4GB or less RAM don't use this parameter or set batch_size to 32 or less
            number_of_noisy_samples - Integer, number of noisy samples generater PER sample in a batch
            training_dataset_path - String
            background_path - String
            valid_paths - String
            testing_paths - String
        '''
        self.learning_mode = learning_mode
        self.sample_rate = sample_rate
        self.background_path = background_path
        self.validation_paths = valid_paths
        self.training_dataset = training_dataset_path
        self.testing_paths = testing_paths
        self.step = step
        self.window_size = window_size
        self.mfcc_dim = mfcc_dim
        self.padd_to_sr = padd_to_sr
        self.spectrogram = spectrogram
        self.batch_size = batch_size
        self.mfcc_features = mfcc_features
        self.generate_noisy_data = generate_noisy_data
        self.number_of_noisy_samples = number_of_noisy_samples
        self.mode = mode

        if self.generate_noisy_data:
            print("WORNING: You are in the noise generating mode. \
                   Your batch size is increased to: {}".format(self.batch_size + self.batch_size * self.number_of_noisy_samples))
        
        self._get_classes_()
        self.id_to_char, self.char_to_id = character_mapper ()

        if mode=='train':
            #Setup everything for the training process
            self._training_files_handler_()
            self._data_ids_()
            self.fit_train()
            self.cur_train_index = 0
            self.cur_valid_index = 0
            self.cur_test_index = 0
        elif mode=='test':
            raise NotImplementedError
        elif mode=='valid':
            raise NotImplementedError
        else:
            raise Exception("Invalid mode.")
          
        
    def _get_classes_(self):
        '''
        Helper functio to get all classes from the dataset.
        '''
        self.classes = os.listdir(self.training_dataset)
        #Finds the longest word/class name in the dataset, this is importatnt for padding and CTC loss
        self.longest_word = len(max(self.classes, key=len))
        
            
    def _data_ids_(self):
        '''
        Gets ids for training, testing and validation sets and mix them for the sampling phase
        '''
        self.training_ids = np.arange(0, len(self.training_files))
        np.random.shuffle(self.training_ids)
        self.valid_ids = np.arange(0, len(self.validation_paths))
        np.random.shuffle(self.valid_ids)
        self.test_ids = np.arange(0, len(self.testing_paths))
        np.random.shuffle(self.test_ids)
        print("Training IDs are ready.")
    
    
    def _training_files_handler_(self):
        '''
        Setups all necessery files for the training process.
        '''
        
        background_files = []
        valid_files = []
        testing_files = []
        training_files = []
        
        #Step 1: Get all background noise files

        for file in os.listdir(self.background_path):
            background_files.append(os.path.join(self.background_path, file))
            
        self.background_noise_files = background_files
        
        #Step 2: Get validation files paths
        
        with open(self.validation_paths, 'r') as f:
            for line in f.readlines():
                valid_files.append(os.path.join(self.training_dataset, line[:-1]))
                
        self.validation_paths = np.array(valid_files)
        
        #Step 3: Get all testing files paths

        with open(self.testing_paths, 'r') as f:
            for line in f.readlines():
                testing_files.append(os.path.join(self.training_dataset, line[:-1]))
                
        self.testing_files = np.array(testing_files)

        #Step 4: Get all training files paths that are not in the validation or testing subsets

        training_files = []
        
        for _class in self.classes:
            for path in glob.glob(self.training_dataset + _class + "/*.wav"):
                if path not in self.validation_paths or path not in self.testing_paths:
                    training_files.append(path)
            
        self.training_files = np.array(training_files)
    
    
    def get_batch(self, batching_mode):
        '''
        Obtain a batch of train, validation, or test data

        :params:
            batching_mode - String, defines models phase (train, test, valid)
        '''

        #Get audio paths and current index based on the batching mode argument
        if batching_mode == 'train':
            audio_paths = self.training_files
            cur_index = self.cur_train_index
        elif batching_mode == 'valid':
            audio_paths = self.training_files
            cur_index = self.cur_valid_index
        elif batching_mode == 'test':
            audio_paths = self.training_files
            cur_index = self.test_valid_index
        else:
            raise Exception("Invalid batching_mode.")

        features = []
        
        #Based on the learning mode setup labels matrix
        if self.learning_mode == 'classification':
            if self.generate_noisy_data and batching_mode != 'train':
                #If the generator is in the noise generation mode generator needs to increase the batch size
                B_SIZE = self.batch_size + (self.batch_size  * self.number_of_noisy_samples)
                labels = np.zeros((B_SIZE, len(self.classes)))
            else:
                labels = np.zeros((self.batch_size, len(self.classes)))
        else:
            if self.generate_noisy_data and batching_mode != 'train':
                B_SIZE = self.batch_size + (self.batch_size  * self.number_of_noisy_samples)
                labels = np.zeros((B_SIZE, self.longest_word))
            else:
                labels = np.zeros((self.batch_size, self.longest_word))
            
        
        
        #Based on the generator state and model phase create batch for input_lenghts, files and labeles lenghts 
        if self.generate_noisy_data and batching_mode != 'train':
            B_SIZE = self.batch_size + (self.batch_size  * self.number_of_noisy_samples)
            batch_files = self.training_files[self.training_ids[cur_index:cur_index+B_SIZE]]
            input_length = np.zeros([B_SIZE, 1])
            label_length = np.zeros([B_SIZE, 1])
        else:
            batch_files = self.training_files[self.training_ids[cur_index:cur_index+self.batch_size]]
            input_length = np.zeros([self.batch_size, 1])
            label_length = np.zeros([self.batch_size, 1])
            

        for p in range(len(batch_files)):
            path = batch_files[p]
            #each sample in the current batch normalize and featurize (generate spectrogram or MFCC features)
            feat = self.normalize(self.featurize(path))
            features.append(feat)
            #Define input lenght based on the features size
            input_length[p] = feat.shape[0]
            
            
            #Calculating labels
            if self.learning_mode == 'classification':

                #For the classification mode, create one_hot_encoded features
                for c in range(len(self.classes)):
                    if self.classes[c] == path.split('/')[-2]:
                        labels[p][c] = 1.0
                        break
            else:

                #For the speech_to_text mode, create int representations of the class/target word
                for c in range(len(self.classes)):
                    if self.classes[c] == path.split('/')[-2]:
                        label = word_to_int(self.classes[c], self.char_to_id)
                        labels[p, :len(label)] = label
                        label_length[p] = len(label)
                        break
            
            #Adding background noise for robustness of the model
            if self.generate_noisy_data and batching_mode == 'train':
                noisy_audios = self.add_noise(path)
                for i in range(len(noisy_audios)):
                    features.append(noisy_audios[i])

        #Convert features to the numpy array for easier manipulation
        X_data = np.array(features)
        
        if self.generate_noisy_data and batching_mode == 'train':
            #repeat alerady calculated data by number of noisy samples plus 1,
            #NOTE: We need to do this so all batch data has equal number of data samples
            labels = np.repeat(labels, self.number_of_noisy_samples + 1,axis=0)
            label_length = np.repeat(label_length, self.number_of_noisy_samples + 1, axis=0)
            input_length = np.repeat(input_length, self.number_of_noisy_samples + 1, axis=0)
        
        # return batch dictionaries based on the learning mdoe
        #NOTE: Keys in dicts should match placeholder names in models that we want to use
        if self.learning_mode == 'classification':
            outputs = {'categorical_crossentropy': np.repeat(np.zeros([self.batch_size]), self.number_of_noisy_samples + 1, axis=0)}
            inputs = {'inputs': X_data, 
                      'labels': labels}
            
            return (inputs, outputs)
        else:
            outputs = {'ctc': np.repeat(np.zeros([self.batch_size]), self.number_of_noisy_samples + 1, axis=0)}
            inputs = {'inputs': X_data, 
                      'labels': labels, 
                      'input_length': input_length, 
                      'label_length': label_length 
                     }
            return (inputs, outputs)   
    
    def add_noise(self, audio_clip):
        '''
        Adds noise to an audio clip.

        :params:
            audio_clip - String, path to the audio clip

        :returns:
            noisy_tracks - Python list, generated nosiy tracks
        '''
        _, audio = wavfile.read(audio_clip)
        
        #padd the track with zeros if the padding mode is True
        if self.padd_to_sr:
            if audio.shape[0] < self.sample_rate:
                audio = np.append(audio, np.zeros(self.sample_rate - audio.shape[0])) 
                
        #randomly choose noisy backgrounds
        noisy_backgrounds = np.random.choice(self.background_noise_files, 
                                            size=self.number_of_noisy_samples)
        
        noisy_tracks = []
        for background in noisy_backgrounds:
            sr, noise = wavfile.read(background)
            #randomly choose starting point of the noisy background
            noisy_sample_start_id = np.random.choice([0, len(noise)-self.sample_rate])
            noisy_sample_end_id = noisy_sample_start_id +self.sample_rate
                
            audio_noise = noise[noisy_sample_start_id:noisy_sample_end_id]

            assert len(audio) == len(audio_noise)
            #Keep 90% of the original audio and add 10% of the noise to it
            #NOTE: These numbers could be randomized as well to generate even more, realistic noise
            new_audio = 0.10 * audio_noise + 0.9 * audio
            noisy_tracks.append(self.featurize(new_audio))
            
        return noisy_tracks
        
    def fit_train(self, k_samples=100):
        """ 
        Estimate the mean and std of the features from the training set

        :params:
            k_samples - Integer, Number of samples used for the estimation process
        """
        k_samples = min(k_samples, len(self.training_files))
        samples = np.random.choice(self.training_files, k_samples)
        feats = [self.featurize(s) for s in samples]
        feats = np.vstack(feats)
        self.feats_mean = np.mean(feats, axis=0)
        self.feats_std = np.std(feats, axis=0)
        print("Normalization parameters are set.")
        
    def next_train(self):
        """ 
        Obtain a batch of training data
        """
        while True:
            ret = self.get_batch('train')
            self.cur_train_index += self.batch_size
            if self.cur_train_index >= len(self.training_files) - self.batch_size:
                self.cur_train_index = 0
                np.random.shuffle(self.training_ids)
            yield ret
            
    def next_valid(self):
        """ 
        Obtain a batch of validation data
        """
        while True:
            ret = self.get_batch('valid')
            self.cur_train_index += self.batch_size
            if self.cur_valid_index >= len(self.validation_paths) - self.batch_size:
                self.cur_train_index = 0
                np.random.shuffle(self.valid_ids)
            yield ret
            
    def next_test(self):
        """ 
        Obtain a batch of test data
        """
        while True:
            ret = self.get_batch('test')
            self.cur_test_index += self.batch_size
            if self.cur_test_index >= len(self.testing_paths) - self.batch_size:
                self.cur_test_index = 0
            yield ret
        
    def featurize(self, audio_clip):
        """ 
        For a given audio clip, calculate the corresponding feature
        :params:
            audio_clip - String, path to the audio clip
        """
        if not isinstance(audio_clip, str):
            audio = audio_clip
        else:
            _, audio = wavfile.read(audio_clip)
        
        #Pad all audios to be the same lengths
        if self.padd_to_sr:
            if audio.shape[0] < self.sample_rate:
                audio = np.append(audio, np.zeros(self.sample_rate - audio.shape[0]))
            
        if self.spectrogram:
            return generate_spectrogram(audio, 
                                       sample_rate=self.sample_rate, 
                                       step_size=self.step, 
                                       window_size=self.window_size)[-1]
        else:
            features = compute_mfcc_features(audio, self.sample_rate, numcep=self.mfcc_dim)
            if len(self.mfcc_features) > 1:
                #This will return delta or delta delta on top of normal mfcc features
                return mfcc_pack(np.array(features)[self.mfcc_features])
            else:
                return features[self.mfcc_features[0]]
            
    def normalize(self, feature, eps=1e-14):
        """ 
        Center a feature using the mean and std
        :params:
            feature - numpy array, Feature to normalize
        """
        return (feature - self.feats_mean) / (self.feats_std + eps)