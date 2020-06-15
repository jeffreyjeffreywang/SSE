from numpy import *
from scipy.signal import *
import yaml
import torch
import numpy as np

def stft( input_seq, dft_size, hop_size=None, zero_pad=0, window=hanning):
    # Set default hop size to half of DFT size
    if hop_size is None:
        hop_size = dft_size/2

    # Make a window
    if window is None:
        w = array( [1])
    elif type(window) is type( input_seq):
        w = window
    else:
        w = window( dft_size+1)[:-1]

    # Forward transform
    if isreal( input_seq).all():
        # Zero pad so that we get full frames
        padding = int( ceil( len(input_seq)/dft_size)*dft_size) - len( input_seq)
        x = pad( input_seq, (0,padding), 'constant')

        # Make the frames and window them
        x = [w*x[i:i+dft_size] for i in range( 0, len(x)-dft_size, hop_size)]

        # Get DFT of each frame in one go
        return fft.rfft( x, n=dft_size+zero_pad, axis=1).T

    # Inverse transform
    else:
        # Inverse FFT and window
        f = w[:,None] * fft.irfft( input_seq, axis=0)

        # Allocate the output
        x = zeros( hop_size*input_seq.shape[1] + dft_size+zero_pad)

        # Overlap add
        for i in range( input_seq.shape[1]):
            x[(hop_size*i):(hop_size*i+dft_size+zero_pad)] += f[:,i]

        # Divide appropriately since we are overlapping frames
        return x * hop_size/dft_size

from scipy.signal import medfilt2d

def denoise( s, n, a, sz=1024, m=0):
    # STFT parameters
    hp = sz//4
    pd = 0
    w = hann( sz+1)[:-1]**.5

    # Get noise spectrum
    fn = mean( abs( stft( n, sz, hp, pd, w)), axis=1)
    
    # Get input STFT
    fs = stft( s, sz, hp, pd, w)

    # Get amplitude and phase
    fa = abs( fs)
    fp = angle( fs)
    
    # Subtract the noise spectrum
    fa -= a * fn[:,None]
    fa = clip( fa, 0, None)
    
    # Clean up a bit
    fa = medfilt2d( fa, m)
   
    # Resynth
    return stft( fa*exp(1j*fp), sz, hp, pd, w)

def add_score(score,scores):
    for key in scores:
        scores[key].append(score[key])

def avg_score(scores):
    avg = {}
    for key in scores:
        avg[key] = sum(scores[key])/len(scores[key])
    return avg

def scores_diff(scores_out,scores_mix):
    diff = {}
    for key in scores_out:
        diff[key] = scores_out[key] - scores_mix[key]
    return diff

# Scale audio to the range between [-0.5,0.5], assume audio is normalized to (0,1)
# Avoid clipping when plotting
def scale_audio(audio):
    r = 0.5/torch.max(torch.abs(audio))
    return r*audio

# Get configuration
def get_config(f):
    with open(f,'r') as s:
        return yaml.load(s)

import matplotlib.pyplot as plt
'''
:param a magnitude spectrogram (numpy.ndarray) of shape (b_size,freq,time)
'''
def plot_spec(a):
    fig = plt.figure(figsize=(15,6))
    plt.subplot(121)
    plt.title('Groundtruth')
    plt.pcolormesh(np.arange(a[0].shape[2]), np.arange(a[0].shape[1]), a[0][0]**.5, cmap=plt.get_cmap('bone_r'))
    plt.colorbar()
        
    plt.subplot(122)
    plt.title('Reconstruction')
    plt.pcolormesh(np.arange(a[1].shape[2]), np.arange(a[1].shape[1]), a[1][0]**.5, cmap=plt.get_cmap('bone_r'))
    plt.colorbar()
    return fig