#%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns

cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)

DATA_DIR = './'


def load_data():
	batch_size = 32
	# not sure what to do with mean and std. copied from example
	norm_mean = (0.1307,)
	norm_std = (0.3081,)

	# cuda is false so will be {}
	kwargs = {'num_workers' : 1, 'pin_memory' : True} if cuda else {}

	# load training data
	train_loader = torch.utils.data.DataLoader(
		datasets.CIFAR10(DATA_DIR, train=True, download=True,
						 transform=transforms.Compose([
							transforms.ToTensor(),
							transforms.Normalize(norm_mean, norm_std)
						 ])),
		batch_size=batch_size, shuffle=False, **kwargs)

	train_loader = torch.utils.data.DataLoader(
		datasets.CIFAR10(DATA_DIR, train=False, transform=transforms.Compose([
							transforms.ToTensor(),
							transforms.Normalize(norm_mean, norm_std)
						 ])),
		batch_size=batch_size, shuffle=False, **kwargs)

if __name__ == '__main__':
	load_data()