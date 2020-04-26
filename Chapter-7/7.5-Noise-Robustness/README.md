# Noise Robustness
Adding noise to the weights can be interpreted as a stochastic implementation of Bayesian inference over the weights. <br>
The Bayesian treatment of learning would consider the model weights to be uncertain and representable via a probability distribution that reflects this uncertainty. <br> <br>
Noise applied to the weights can also be interpreted as equivalent (under some assumptions) to a more traditional form of regularization, encouraging stability of the function to be learned. <br><br>
For more details about the theory, see paragraph 7.5 from [Chapter 7 - Regularization](https://www.deeplearningbook.org/contents/regularization.html)

## · Main code taken from [Basic MNIST Example](https://github.com/pytorch/examples/tree/master/mnist)

## · Requirements
```bash
pip install -r requirements.txt
```
## · Run experiments:
### 1. Train without adding noise to the parameters
```bash
python train.py --train-eta 0.0 --test-eta 0.1
```

### 2. Train with added noise to the parameters
```bash
python train.py --train-eta 0.1 --test-eta 0.1
```

------------------------------------------------------------------------

## Notes
* Average meter. The original implementation can be found [here](https://github.com/pytorch/examples/blob/master/imagenet/main.py#L359-L380) 
* When training with `-train-eta 0.0` the noisy and not noisy accuracy and losses will be the same at trainig time. <br> Expect a low accuracy at test time with the noisy model
