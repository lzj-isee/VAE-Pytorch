
## VAE Pytorch Implementation
This is the Pytorch implementation of variational auto-encoder, applying on MNIST dataset.

### Usage
```python
$ python train.py
```
The code is self-explanatory, you can specify the num_epoch you want in the function `run()`.

### Result
I just trained the VAE ~ 30 epochs. Here are some visualization result.

- reconstruction result (1,10,20 respectively)  

![](./result/reconstructed-1.png)

![](./result/reconstructed-10.png)

![](./result/reconstructed-20.png)

- randomly generated digits (1,10,20 respectively)  

![](./result/random_sampled-1.png)

![](./result/random_sampled-10.png)

![](./result/random_sampled-20.png)

