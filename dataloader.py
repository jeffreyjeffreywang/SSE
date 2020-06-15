import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils import data
from scipy.io.wavfile import read
from librosa.core import resample

import os
import random
START = 5 # audio files start at 5th second
LEN = 2 # sample 2 sec clip
EPS = 1e-8
TEST_SIZE = 50

def get_files():
    # Generate speaker id list
    speaker_gender = ['f','m']
    speaker_num = list(range(1,11))
    speakers = []
    for gender in speaker_gender:
        for num in speaker_num:
            speakers.append(gender+str(num))

    # Randomly sample 6 speakers and 2 sentences for train data
    # Partition speakers to disjoint sets: train_A, train_B, test_A, test_B
    perm = np.random.permutation(range(20)).tolist()
    train_A_speaker_num = [speakers[i] for i in perm[:8]]
    train_B_speaker_num = [speakers[i] for i in perm[8:16]]
    test_A_speaker_num = [speakers[i] for i in perm[16:16]]
    test_B_speaker_num = [speakers[i] for i in perm[16:]]

    # Partition scripts to disjoint sets: train_A, train_B, test
    perm = np.random.permutation(range(1,6)).tolist()
    train_A_script_num = perm[:2]
    train_B_script_num = perm[2:4]
    test_script_num = perm[4]

    # Domain A training and testing files
    train_A_files = []
    test_A_files = []
    for speaker in train_A_speaker_num:
        for script in train_A_script_num:
            train_A_files.append('{}_script{}'.format(speaker,script))
    for speaker in test_A_speaker_num:
        test_A_files.append('{}_script{}'.format(speaker,test_script_num))
    # Domain B training and testing files
    train_B_files = []
    test_B_files = []
    for speaker in train_B_speaker_num:
        for script in train_B_script_num:
            train_B_files.append('{}_script{}'.format(speaker,script))
    for speaker in test_B_speaker_num:
        test_B_files.append('{}_script{}'.format(speaker,test_script_num))
        
    return train_A_files, train_B_files, test_A_files, test_B_files

# Get noise files used to generate mixtures
def get_all_noise_files(dataset='BBC.16K',num_noise_files=1,city='London'):
    if dataset == 'BBC.16K': # use Ambience
        root_dir = '/mnt/data/Sound Effects/BBC.16k'
        ambience_files = ['{}/{}'.format(root_dir,i) for i in os.listdir(root_dir) if i.startswith('Ambience'+city)] # Ambience
        random.shuffle(ambience_files)
        files = {}
        files[0] = ambience_files[:num_noise_files]
    return files

def get_noise_files(all_noise_files,noise_class_ids):
    noise_files = []
    for c in noise_class_ids:
        noise_files += all_noise_files[c]
    random.shuffle(noise_files)
    return noise_files, noise_files

# Dataset for DAPS
class Daps(data.Dataset):
    '''
    :param version - A list of versions. If there are two elements in the list, 
                                         the first element is the noisy version and the second is the clean version.
    '''
    def __init__(self,version,files,sr,clip_samples,pure_noise,flag):
        self.version = version
        self.root_dir = '/mnt/data/daps/'
        self.files = files
        self.sr = sr 
        self.clip_samples = clip_samples
        self.threshold = 12
        self.size = 1024
        self.hop = 256
        self.pure_noise = pure_noise
        self.flag = flag
    
    def __getitem__(self,index):
        while True:
            notnoise = 1
            # Randomly sample a file
            f = random.choice(self.files)
            fs, audio = read('{}{}/{}_{}.wav'.format(self.root_dir,self.version[0],f,self.version[0]))
            audio = audio.astype('float32')
            # Randomly sample a clip
            r = random.random()
            is_silence = False
            if r < self.pure_noise and self.flag == 'train':
                start = random.randint(0, START*fs-LEN*fs)
                is_silence = True
                notnoise = 0
            else: 
                start = random.randint(START*fs,len(audio)-LEN*fs)
            # Resample the clip
            clip = resample(audio[start:start+LEN*fs],fs,self.sr) / 1e5
            # Thresholding: discard clip if the clip contains too much silence
            if not is_silence and np.sum(clip**2) < self.threshold:
                continue
            # Normalize the clip
            mu, sigma = np.mean(clip), np.std(clip)
            normalized = torch.from_numpy((clip-mu)/sigma)

            if len(self.version) > 1:
                fs, audio_clean = read('{}{}/{}_{}.wav'.format(self.root_dir,self.version[1],f,self.version[1]))
                audio_clean = audio_clean.astype('float32')
                # Extract the corresponding clean clip
                if is_silence:
                    normalized_clean = torch.zeros(LEN*self.sr).float()
                else:
                    clip_clean = resample(audio_clean[start:start+LEN*fs],fs,self.sr)
                    mu_clean, sigma_clean = np.mean(clip_clean), np.std(clip_clean)
                    normalized_clean = torch.from_numpy((clip_clean-mu_clean)/sigma_clean)
                
                if self.flag == 'train':
                    return normalized, normalized_clean, notnoise
                else:
                    return normalized, normalized_clean
            
            return normalized

    def __len__(self):
        return 1000 # sentinel value

import soundfile as sf
# Dataset for custom noises
class DapsNoise(data.Dataset):
    def __init__(self,clean_files,noise_files,sr,clip_samples,pure_noise,snr,flag):
        self.clean_root_dir = '/mnt/data/daps/'
        self.clean_files = clean_files
        self.noise_files = noise_files
        self.sr = sr
        self.clip_samples = clip_samples
        self.threshold = 12
        self.pure_noise = pure_noise
        self.snr = snr
        self.flag = flag
        
    def __getitem__(self,index):
        while True:
            notnoise = 1
            # Clean files
            if len(self.clean_files) != 0:
                # Randomly sample a clean file
                f = random.choice(self.clean_files)
                fs,audio = read('{}{}/{}_{}.wav'.format(self.clean_root_dir,'clean',f,'clean'))
                audio = audio.astype('float32')
                # Randomly sample a clean clip
                r = random.random()
                if r < self.pure_noise and self.flag == 'train':
                    normalized_clean = torch.zeros(LEN*self.sr).float()
                    notnoise = 0
                else: 
                    start = random.randint(START*fs,len(audio)-LEN*fs)
                    clip = resample(audio[start:start+LEN*fs],fs,self.sr)/1e5

                    if r >= self.pure_noise and np.sum(clip**2) < self.threshold and self.flag == 'train':
                        continue
                    mu, sigma = np.mean(clip), np.std(clip)
                    normalized_clean = torch.from_numpy((clip-mu)/sigma)
                
            # Noise files
            if len(self.noise_files) != 0:
                nf = random.choice(self.noise_files)
                audio_noise, fs = sf.read(nf)
                if len(audio_noise.shape) > 1:
                    audio_noise = np.mean(audio_noise,axis=1)
                audio_noise = audio_noise.astype('float32')
                # Randomly sample a clip of noise
                if len(audio_noise) < LEN*fs: continue
                start = random.randint(0,len(audio_noise)-LEN*fs)
                clip_noise = resample(audio_noise[start:start+LEN*fs],fs,self.sr)
                mu_noise, sigma_noise = np.mean(clip_noise), np.std(clip_noise)
                normalized_noise = torch.from_numpy((clip_noise-mu_noise)/(sigma_noise+EPS))
                
                # Mix the noise with the clean audio clip at given SNR level
                interference = 10**(-self.snr/20)*normalized_noise
                if r < self.pure_noise and self.flag == 'train':
                    mixture = interference
                else:
                    mixture = normalized_clean + interference
                mu_mixture, sigma_mixture = torch.mean(mixture), torch.std(mixture)
                mixture = (mixture-mu_mixture) / sigma_mixture 

            if len(self.noise_files) != 0:
                if self.flag == 'train':
                    return mixture, normalized_clean, notnoise 
                if self.flag == 'test':
                    return mixture, normalized_clean
            return normalized_clean

    def __len__(self):
        return 1000 # sentinel value

# Get the dataloader for clean, mix, and test
def get_train_test_data(config,train_A_files,train_B_files,test_B_files,train_noise_files=None,test_noise_files=None):
    if config['urban_noise']:
        # Clean
        train_A_data = DapsNoise(train_A_files,[],config['sr'],config['clip_size'],config['pure_noise_a'],config['snr'],'train')
        # Noisy train
        train_B_data = DapsNoise(train_B_files,train_noise_files,config['sr'],config['clip_size'],\
                                                    config['pure_noise_b'],config['snr'],'train')
        # Noisy test
        test_B_data = DapsNoise(test_B_files,test_noise_files,config['sr'],config['clip_size'],\
                                                    config['pure_noise_b'],config['snr'],'test')
    else:
        # Training data
        train_A_data = Daps([config['version_A']], train_A_files, config['sr'], \
                            config['clip_size'], config['pure_noise_a'],'train')
        train_B_data = Daps([config['version_B'],config['version_A']], train_B_files, config['sr'], \
                            config['clip_size'], config['pure_noise_b'],'train')
        # Testing data
        test_B_data = Daps([config['version_B'],config['version_A']], test_B_files, \
                                  config['sr'], config['clip_size'], config['pure_noise_b'],'test')

    train_A_dataloader = DataLoader(train_A_data, batch_size=config['b_size'], shuffle=True, \
                                    num_workers=config['num_workers'], drop_last=True)
    train_B_dataloader = DataLoader(train_B_data, batch_size=config['b_size'], shuffle=True, \
                                    num_workers=config['num_workers'], drop_last=True)
    test_B_dataloader = DataLoader(test_B_data, batch_size=1, shuffle=True)
    
    test_B_data = []
    for i, audio_pair in enumerate(test_B_dataloader):
        if i >= TEST_SIZE: break
        test_B_data.append(audio_pair)
    return train_A_dataloader, train_B_dataloader, test_B_data