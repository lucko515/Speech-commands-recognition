from python_speech_features import mfcc
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal

import numpy as np
import librosa
import glob
import os



def generate_spectrogram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10, log=True):
    '''
    Generates a spectrogram based on the input audio signal.

    :params:
        audio - numpy array, raw audio signal (example: in wav format)
        sample_rate - Integer
        window_size - Integer, number of milliseconds that we will consider at the time
        step_size - Integer, number of milliseconds (points) to move the kernel,
                    if the step_size == window_size there is no overlapping beteween signal segments
        eps - Integer, very small number used to help calculating log values of the spectrogram
        log - Boolean, if True, this function will return log values of the specgora

    :returns:
        freq - array of sample frequencies.
        times - array of segment times.
        spec - numpy matrix, spectrogram of the input audio signal
    '''
    
    #Calculates the length of each segment
    nperseg = int(round(window_size * sample_rate / 1e3))
    #Calculates the number of points to overlap between segments
    noverlap = int(round(step_size * sample_rate / 1e3))
    
    #Computes spectrogram
    freqs, times, spec = signal.spectrogram(audio, 
                                            fs=sample_rate, 
                                            window='hann', 
                                            nperseg=nperseg, 
                                            noverlap=noverlap, 
                                            detrend=False)
    
    if log:
        return freqs, times, np.log(spec.T.astype(np.float32) + eps)
    
    return freqs, times, spec


def compute_mfcc_features(audio_sample, sample_rate, numcep=13):
    '''
    Computes MFCC, delta MFCC and delta-delta MFCC features of the input audio sample.
    
    :params:
        audio_sample, numpy array, raw audio signal (example: in wav format)
        sample_rate, Integer
        numcap - Integer, the number of cepstrum features to return

    :returns:
        mfcc_features, Numpy array, 
        delta1_mfcc, numpy array,
        delta2_mfcc, numpy array, 
    '''
    
    mfcc_features = mfcc(audio_sample, sample_rate, numcep=numcep)

    delta1_mfcc = librosa.feature.delta(mfcc_features, order=1)

    delta2_mfcc = librosa.feature.delta(mfcc_features, order=2)
        
    return mfcc_features, delta1_mfcc, delta2_mfcc


def audio_resampler(audio_sample, new_sample_rate, old_sample_rate):
    '''
    Resamples an audio signal to a new sample rate.

    :params:
        audio_sample - numpy array, raw audio signal (example: in wav format)
        new_sample_rate - Integer
        old_sample_rate - Integer

    :returns:
        resampled_signal - numpy array, resampled audio signal to new sample rate
    '''
    resampled_signal = signal.resample(audio_sample, int(new_sample_rate/new_sample_rate * audio_sample.shape[0]))
    return resampled_signal


def mfcc_pack(features):
    '''
    Packs all MFCC features into one single input matrix.

    :params:
        features - List of all MFCC features that are being packed. Example: [mfcc, delta1_mfcc]

    :returns:
        unified numpy matrix made of all MFCC features
    '''
    return np.hstack(features)