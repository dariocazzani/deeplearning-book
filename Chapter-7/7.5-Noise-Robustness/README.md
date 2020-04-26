# Main code taken from [Basic MNIST Example](https://github.com/pytorch/examples/tree/master/mnist)

```bash
pip install -r requirements.txt
```

### Train without adding noise to the parameters
python train.py --train-eta 0.0 --test-eta 0.1

### Train with added noise to the parameters
python train.py --train-eta 0.1 --test-eta 0.1

### Notes
* Average meter. The original implementation can be found [here](https://github.com/pytorch/examples/blob/master/imagenet/main.py#L359-L380) 
* When training with `-train-eta 0.` the noisy and not noisy accuracy and losses will be the same at trainig time.