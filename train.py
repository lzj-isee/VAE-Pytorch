# -*- coding: utf-8 -*-
"""
    @Author  : LiuZhian
    @Time    : 2019/10/5 0005 下午 10:24
    @Comment : 
"""

import torch
import torch.optim as  optim
from torchvision.utils import save_image
from VAE import *
from utils import prepare_MNIST
import os

BATCH_SIZE = 128
NUM_WORKERS = 2
H_DIM = 200
Z_DIM = 20
LEARNING_RATE = 1e-3
INPUT_DIM = 28 * 28

def run(num_epochs=10):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	myVAE = VAE(INPUT_DIM, H_DIM, Z_DIM).to(device)
	optimizer = optim.Adam(myVAE.parameters(), lr=LEARNING_RATE)

	trainloader, testloader, classes = prepare_MNIST(BATCH_SIZE, NUM_WORKERS)
	for epoch in range(num_epochs):
		inputs = None
		for i, data in enumerate(trainloader):
			# get the inputs; data is a list of [inputs, labels]
			# Remember to deploy the input data on GPU
			inputs = data[0].to(device).view(-1, INPUT_DIM)

			# forward
			res, mu, log_sigma = myVAE(inputs)

			# Calculate the loss. Note that the loss includes two parts.
			# 1. the reconstruction loss.
			# We regard the MNIST as binary classification
			reconstruction_loss = F.binary_cross_entropy(res, inputs, size_average=False)

			# 2. KL-divergence
			# D_KL(Q(z|X,y) || P(z|X)); calculate in closed form as both dist. are Gaussian
			divergence = 0.5 * torch.sum(torch.exp(log_sigma) + torch.pow(mu, 2) - 1. - log_sigma)

			loss = reconstruction_loss + divergence

			# zero out the paramter gradients
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# print statistics every 100 batches
			if (i + 1) % 100 == 0:
				print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
					  .format(epoch + 1, num_epochs, i + 1, len(trainloader), reconstruction_loss.item(),
							  divergence.item()))

		if not os.path.exists("./result"):
			os.makedirs("./result")
		# do test after each epoch
		with torch.no_grad():
			# show the reconstruction image
			res, _, _ = myVAE(inputs)
			x_concat = torch.cat([inputs.view(-1, 1, 28, 28), res.view(-1, 1, 28, 28)], dim=3)
			save_image(x_concat, ("./result/reconstructed-%d.png" % (epoch + 1)))

			# we randomly sample some images' latent vectors from its distribution
			z = torch.randn(BATCH_SIZE, Z_DIM).to(device)
			res = myVAE.decode(z).view(-1, 1, 28, 28)
			save_image(res, "./result/random_sampled-%d.png" % (epoch + 1))


if __name__ == '__main__':
	run(30)
