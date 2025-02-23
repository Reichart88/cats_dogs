# Cats vs Dogs Classifier

Binary image classifier built with MobileNet architecture to distinguish between cats and dogs. Features transfer learning approach and data augmentation for improved accuracy.

## Features
- MobileNet with pre-trained ImageNet weights
- Custom data augmentation pipeline
- Two-stage model training:
  - Feature extraction using MobileNet
  - Fine-tuning with custom layers
- Training visualization with accuracy/loss plots

## Dataset
Uses the popular Cats and Dogs dataset with:
- 6000 training images
- 1000 validation images
- 1005 test images

