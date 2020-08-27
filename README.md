# SSE
Code for Self-supervised Speech Enhancement network

# Disclaimer

University of Illinois

Open Source License

Copyright © 2020, University of Illinois at Urbana Champaign. All rights reserved.

Developed by:

Yu-Che Jeffrey Wang, Shrikant Venkataramani, Paris Smaragdis

University of Illinois at Urbana-Champaign, Adobe Research

This work was supported by NSF grant 1453104.

Paper link:
https://arxiv.org/pdf/2006.10388.pdf

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal with the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimers in the documentation and/or other materials provided with the distribution.
Neither the names of Computational Audio Group, University of Illinois at Urbana-Champaign, nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission.
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.
———————————————————

### Software requirements: 
* Python v > 3.5
* Numpy v > 1.12 
* Pytorch v > 0.1 
* wandb v == 0.9.0
* SoundFile v ==0.10.2
* librosa v ==0.7.2

### Links
The paper is available [here]

Speech enhancement examples are available [here]

### Commands
Run the following commands in the terminal for running the code:

python3 train.py --urban_noise False (speech enhancement experiments on the DAPS dataset)

python3 train.py --urban_noise True (speech enhancement experiments on the BBC dataset)

### Brief Description
Supervised learning for single-channel speech enhancement requires carefully labeled training examples where the noisy mixture is input into the network and the network is trained to produce an output close to the ideal target. To relax the conditions on the training data, we consider the task of training speech enhancement networks in a self-supervised manner. We first use a limited training set of clean speech sounds and learn a meaningful latent representation by autoencoding on their magnitude spectrograms. We then autoencode on speech mixtures recorded in noisy environments and train the resulting autoencoder to share a latent representation with the clean examples. We show that using this training schema, we can now map noisy speech to its clean version using a network that is autonomously trainable without requiring labeled training examples or human intervention. 
