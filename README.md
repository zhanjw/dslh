# Description
Implementation of DSLH Algorithms. ICIP2020

# How to run
Firstly, generate the sample databases owned by each class
```
pyhon preprocess.py
```
According to your data storage location, adjust the directory and hyperparameters in config.py
Then, you can easily train and test just by
```
pyhon main.py
```

# Dataset:
CIFAR10 (Lossless PNG format):
https://drive.google.com/open?id=1NZ5QKW2zqzN-RQ4VDpuOAb-UgcsTPUJK
Size: 140M

NUS-WIDE:
https://drive.google.com/open?id=0B7IzDz-4yH_HMFdiSE44R1lselE
Size: 5G

ImageNet:
https://drive.google.com/open?id=0B7IzDz-4yH_HSmpjSTlFeUlSS00
Size: 14G

COCO:
https://drive.google.com/open?id=0B7IzDz-4yH_HN0Y0SS00eERSUjQ
Size: 19G

# Plaform information:
python                    3.6.8
numpy                     1.16.5
pytorch                   1.0.0
torchvision               0.2.2
pillow                    6.2.0