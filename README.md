# CondenseNet tensorflow
Tensorflow implementation of [CondenseNet: An Efficient DenseNet using Learned Group Convolutions](https://arxiv.org/abs/1711.09224). The code is tested with cifar10, *inference phase not implemented yet*.

![Model architecture](https://i.imgur.com/f98IK2e.png)

Official PyTorch implementation by @ShichenLiu [here](https://github.com/ShichenLiu/CondenseNet).

## Prerequisites
- Python 2.7+ (3.5+ is recommended)
- [NumPy](http://www.numpy.org/)
- [TensorFlow 1.0+](https://www.tensorflow.org/)


## Data
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)


## Preparation
- Go to `data/` folder and run `python2 generate_cifar10_tfrecords.py --data-dir=./cifar-10-data`. This code is directly borrowed from tensorflow official repo and have to be run with python 2.7+.


## Train
Use default parameters:
```bash
python main.py
```
Check out tunable hyper-parameters:
```bash
python main.py --help
```
Other parameters including `stages, groups, condense factor, and growth rate` are in `experiment.py`.

## Notes
- Training for 300 epochs with the default settings reach testing accuracy 93.389% (paper report is 94.94%). There might be some details I didn't notice, feel free to point them out.
- All the default parameters settings follow the paper/official pytorch implementation.
- Current implmentations of standard group convolution and learned group convolution are very inefficient (a bunch of reshape, transpose and concat), looking for help to build much more efficient graph.
- Evaluation phase (index select) has not been implemented yet, looking for potential help as well :D.
- Issues are welcome!


## Resources
- [The paper](https://arxiv.org/abs/1711.09224).
- [Official PyTorch Implementation](https://github.com/ShichenLiu/CondenseNet).
