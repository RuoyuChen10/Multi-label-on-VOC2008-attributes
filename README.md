# Multi label learning on VOC2008 attributes

## 1. Datasets

VOC2008: [https://www.kaggle.com/sulaimannadeem/pascal-voc-2008](https://www.kaggle.com/sulaimannadeem/pascal-voc-2008)

Attribute dataset: [https://vision.cs.uiuc.edu/attributes/](https://vision.cs.uiuc.edu/attributes/)

## 2. Configuration

see configuration file in `./configs/Base-ResNet101-B.yaml`.

## 3. Train model

```shell
python train.py --configuration-file ./configs/Base-ResNet101-B.yaml --gpu-device 0,1
```

## 4. Evaluation

```shell
python evaluation.py \
    --configuration-file ./configs/Base-ResNet101-B.yaml\
    --gpu-device 0,1
    --Test-model <path to your model>
```