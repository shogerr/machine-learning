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
IMG_WIDTH = 32
IMG_HEIGHT = 32


class Net(nn.Module):
	# should play with out_features and dropout levels
	def __init__(self):
		super(Net, self).__init__()
		# nn.Linear(in_features, out_features)
		self.fc1 = nn.Linear(IMG_WIDTH*IMG_HEIGHT, 50)
		# nn.Dropout(probability dropout i think)
		self.fc1_drop = nn.Dropout(0.2)
		self.fc2 = nn.Linear(50, 50)
		self.fc2_drop = nn.Dropout(0.2)
		self.fc3 = nn.Linear(50, 10)

	# not sure what this does yet. I think it traverses through NN?
	# backward method is auto generated
	def forward(self, x):
		x = x.view(-1, IMG_WIDTH*IMG_HEIGHT)
		x = F.relu(self.fc1(x))
		x = self.fc1_drop(x)
		x = F.relu(self.fc2(x))
		x = self.fc2_drop(x)
		return F.log_softmax(self.fc3(x))


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
		batch_size=batch_size, shuffle=True, **kwargs)

	# load training data
	validation_loader = torch.utils.data.DataLoader(
		datasets.CIFAR10(DATA_DIR, train=False, transform=transforms.Compose([
							transforms.ToTensor(),
							transforms.Normalize(norm_mean, norm_std)
						 ])),
		batch_size=batch_size, shuffle=False, **kwargs)

	for (X_train, y_train) in train_loader:
		print('X_train:', X_train.size(), 'type:', X_train.type())
		print('y_train:', y_train.size(), 'type:', y_train.type())
		break

	return train_loader, validation_loader

def train(epoch, model, train_loader, optimizer, log_interval=100):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		if cuda:
			data, target = data.cuda(), target.cuda()

		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.data[0]))

if __name__ == '__main__':
	train_loader, validation_loader = load_data()

	model = Net()
	if cuda:
		model.cuda()

	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

	print(model)

	train(1, model, train_loader, optimizer)
