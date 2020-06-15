from model import *
from dataloader import *
from metrics import *
from utils import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import wandb
import os
import argparse
import yaml
import pickle
import time

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--config_file',type=str, default='config.yaml')
parser.add_argument('--pretrain_clean', type=bool, default=False)
parser.add_argument('--pretrain_CAE', type=str, default='cae.pkl')
parser.add_argument('--pretrain', type=bool, default=False)
parser.add_argument('--pretrain_MAE', type=str)
parser.add_argument('--urban_noise', type=bool, default=False)
args = parser.parse_args()

# Directory to store all the models
BASE_DIR = '.'

# Select device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get configuration
config = get_config(args.config_file)
config['urban_noise'] = args.urban_noise
    
# Setup clean, mix, test data
train_A_files, train_B_files, test_A_files, test_B_files = get_files()
train_noise_files, test_noise_files = get_noise_files(get_all_noise_files(config['noise_dataset'],config['num_noise_files'],config['city']),config['noise_class_ids'])

train_A_dataloader, train_B_dataloader, test_B_data = get_train_test_data(config,train_A_files,train_B_files,test_B_files,train_noise_files,test_noise_files)

# Setup model
if args.pretrain:
    with open(args.pretrain_MAE, 'rb') as input:
        trainer = pickle.load(input)
elif args.pretrain_clean:
    with open(args.pretrain_CAE, 'rb') as input:
        vae_clean = pickle.load(input)
        trainer = SSE(config,vae_clean).to(device)
else:
    trainer = SSE(config).to(device)
    
stft = STFT(filter_length=config['filter_length'],hop_length=config['hop_length'],\
            win_length=config['win_length'],window=config['window']).to(device)
    
# Save model
if not os.path.exists('{}'.format(BASE_DIR)):
    os.makedirs('{}'.format(BASE_DIR))
with open('{}/sep_trainer_init.pkl'.format(BASE_DIR), 'wb') as output:
    pickle.dump(trainer, output, pickle.HIGHEST_PROTOCOL)
# Save config
with open('{}/config_.yaml'.format(BASE_DIR), 'w') as f:
    data = yaml.dump(config, f)
    
start_train = time.time()
wandb.init(anonymous='allow', project='audio_sep')

if not args.pretrain_clean:
    # Train the clean autoencoder
    print('Start training autoencoder A')
    for epoch in range(1,config['epochs_a']+1):
        start = time.time()
        for i, audio in enumerate(train_A_dataloader):
            loss, x_a_recon = trainer(audio.float().to(device),'a')
            wandb.log({'train_loss': loss})
        if epoch % 10 == 0:
            end = time.time()
            # Domain A original audio
            wandb.log({"audio_a": [wandb.Audio(scale_audio(audio[0].cpu()).numpy(), \
                                                 caption="input_audio_a", sample_rate=config['sr'])]})
            # Domain A within domain reconstruction
            wandb.log({"recon_audio_a": [wandb.Audio(scale_audio(x_a_recon[0].detach().cpu()).numpy(), \
                                                 caption="recon_audio_a", sample_rate=config['sr'])]})
            print('Epoch %d -- loss: %.3f, time: %.3f'%(epoch,torch.mean(loss.detach()).item(),end-start))

    print('Finish training autoencoder A')

    with open(args.pretrain_CAE, 'wb') as output:
        pickle.dump(trainer.gen_a, output, pickle.HIGHEST_PROTOCOL)

# Train the noisy autoencoder
print('Start training autoencoder B')
for epoch in range(1,config['epochs_b']+1):
    start = time.time()
    for i, (audio_b,audio_b_clean,notnoise) in enumerate(train_B_dataloader):
        loss,mag_b,mag_b_recon,mag_ba,x_b_recon,x_ba = trainer(audio_b.float().to(device),'b',notnoise.float().to(device))
        wandb.log({'train_loss': loss})
        if i == len(train_B_dataloader)-1 and epoch%10 == 0:
            end = time.time()
            # Log losses on the terminal
            print('Epoch %d -- loss: %.3f, time: %.3f'%(epoch, torch.mean(loss.detach()).item(),end-start))
            
            # Evaluation
            scores_out = {'csig':[],'cbak':[],'covl':[],'pesq':[],'ssnr':[]}
            scores_mix = {'csig':[],'cbak':[],'covl':[],'pesq':[],'ssnr':[]}
            for i, (audio_b_test,audio_b_clean_test) in enumerate(test_B_data):
                x_b_recon,mag_ba,x_ba,mag_b = trainer(audio_b_test.float().to(device),'eval')
                mag_ba_gt,_ = stft(audio_b_clean_test.float().to(device))
                add_score(eval_composite(audio_b_clean_test[0,:].float().numpy(),\
                                         x_ba[0,:].detach().cpu().float().numpy()),scores_out)
                add_score(eval_composite(audio_b_clean_test[0,:].float().numpy(),\
                                         audio_b_test[0,:].detach().cpu().float().numpy()),scores_mix)
                if i < 5:
                    # Domain B original audio
                    wandb.log({"audio_b_test-{}".format(i): [wandb.Audio(scale_audio(audio_b_test[0].cpu()).numpy(), \
                                                     caption="input_audio_b_test-{}".format(i), sample_rate=config['sr'])]})
                    # Domain B within domain reconstruction
                    wandb.log({"recon_audio_b_test-{}".format(i): [wandb.Audio(scale_audio(x_b_recon[0].detach().cpu()).numpy(), \
                                                 caption="recon_audio_b_test-{}".format(i), sample_rate=config['sr'])]})
                    # Domain B -> domain A cross domain reconstruction
                    wandb.log({"recon_audio_ba_test-{}".format(i): [wandb.Audio(scale_audio(x_ba[0].detach().cpu()).numpy(), \
                                                     caption="recon_audio_ba_test-{}".format(i), sample_rate=config['sr'])]})
                    # Domain B corresponding clean version
                    wandb.log({"audio_ba_groundtruth_test-{}".format(i): \
                               [wandb.Audio(scale_audio(audio_b_clean_test[0].cpu()).numpy(),\
                                caption="audio_ba_groundtruth_test-{}".format(i), sample_rate=config['sr'])]})
                    fig_ba = plot_spec([mag_ba_gt.detach().cpu().numpy(),mag_ba.detach().cpu().numpy()])
                    wandb.log({"spec_ba_test-{}".format(i): [wandb.Image(fig_ba, caption='spectrogram_ba_test-{}'.format(i))]})
            
            # Save model periodically
            if epoch%100 == 0:
                with open('{}/sep_trainer_ep{}.pkl'.format(BASE_DIR,epoch), 'wb') as output:
                    pickle.dump(trainer, output, pickle.HIGHEST_PROTOCOL)
                avg_score_mix = avg_score(scores_mix)
                avg_score_out = avg_score(scores_out)
                print('Mix: ', avg_score_mix)
                print('Out: ', avg_score_out)
                    
print('Finish training autoencoder B')
end_train = time.time()
print('Total training time: %.3f'%(end_train-start_train))

# Save model
with open('{}/sep_trainer_final.pkl'.format(BASE_DIR), 'wb') as output:
    pickle.dump(trainer, output, pickle.HIGHEST_PROTOCOL)
