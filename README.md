# Convolutional 2D KAN Implementation

This repositry contains 2 drop-in convolutional KAN replacements. Each work on top of a different KAN implementation:

1. [Efficient implementation of Kolmogorov-Arnold Network (KAN)](https://github.com/Blealtan/efficient-kan) using ```ConvKAN.Version.ORIGINAL```
2. [Original KAN implementation](https://github.com/KindXiaoming/pykan) using ```ConvKAN.Version.EFFICIENT```

# Installation
```bash
git clone git@github.com/omarrayyann/KAN-Conv2D
cd KAN-Conv2D
pip install -r requirements.txt
```

# Usage

You should be able to just replace ```torch.nn.Conv2D()``` with ```ConvKAN()```

```python3

from ConvKAN import ConvKAN

# Implementation built on the efficient KAN (https://github.com/KindXiaoming/pykan)
conv = ConvKAN(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1, version=ConvKAN.Version.EFFICIENT)

# Implementation built on the original KAN (https://github.com/KindXiaoming/pykan)
conv = ConvKAN(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1, version=ConvKAN.Version.ORIGINAL)

```
