# Multi label learning on VOC2008 attributes

Chinese can refer my csdn blog [https://blog.csdn.net/Exploer_TRY/article/details/118910514?spm=1001.2014.3001.5501](https://blog.csdn.net/Exploer_TRY/article/details/118910514?spm=1001.2014.3001.5501) for more information.

## 1. Datasets

VOC2008: [https://www.kaggle.com/sulaimannadeem/pascal-voc-2008](https://www.kaggle.com/sulaimannadeem/pascal-voc-2008)

Attribute dataset: [https://vision.cs.uiuc.edu/attributes/](https://vision.cs.uiuc.edu/attributes/)

> @inproceedings{farhadi2009describing,\
    title={Describing objects by their attributes},\
    author={Farhadi, Ali and Endres, Ian and Hoiem, Derek and Forsyth, David},\
    booktitle={2009 IEEE conference on computer vision and pattern recognition},\
    pages={1778--1785},\
    year={2009},\
    organization={IEEE}\
}

## 2. Configuration

see configuration file in `./configs/Base-ResNet101-B.yaml`.

## 3. Network

You can download pretrained model, or get other network model from here: [https://github.com/pytorch/vision/tree/master/torchvision/models](https://github.com/pytorch/vision/tree/master/torchvision/models)

For example, in this project, we adopt resnet101, can download from [https://download.pytorch.org/models/resnet101-63fe2227.pth](https://download.pytorch.org/models/resnet101-63fe2227.pth), then in configuration file, change the variable `PRETRAINED`.

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

## 5. Example results

```shell
+-----------------+--------+-------+--------+--------+-----------+--------+--------+
|  Attribute Name |   TP   |   FN  |   FP   |   TN   | Precision | Recall |  ACC   |
+-----------------+--------+-------+--------+--------+-----------+--------+--------+
|    0. 2D Boxy   | 298.0  |  38.0 | 907.0  | 5112.0 |   0.2473  | 0.8869 | 0.8513 |
|    1. 3D Boxy   | 837.0  |  88.0 | 868.0  | 4562.0 |   0.4909  | 0.9049 | 0.8496 |
|     2. Round    |  47.0  |  20.0 | 635.0  | 5653.0 |   0.0689  | 0.7015 | 0.8969 |
|   3. Vert Cyl   | 372.0  |  40.0 | 263.0  | 5680.0 |   0.5858  | 0.9029 | 0.9523 |
|   4. Horiz Cyl  | 229.0  |  31.0 | 455.0  | 5640.0 |   0.3348  | 0.8808 | 0.9235 |
|   5. Occluded   | 1817.0 | 966.0 | 1623.0 | 1949.0 |   0.5282  | 0.6529 | 0.5926 |
|     6. Tail     | 445.0  |  72.0 | 524.0  | 5314.0 |   0.4592  | 0.8607 | 0.9062 |
|     7. Beak     | 163.0  |  46.0 |  79.0  | 6067.0 |   0.6736  | 0.7799 | 0.9803 |
|     8. Head     | 2778.0 | 225.0 | 620.0  | 2732.0 |   0.8175  | 0.9251 | 0.8670 |
|      9. Ear     | 1865.0 | 182.0 | 915.0  | 3393.0 |   0.6709  | 0.9111 | 0.8274 |
|    10. Snout    | 588.0  |  32.0 | 216.0  | 5519.0 |   0.7313  | 0.9484 | 0.9610 |
|     11. Nose    | 1434.0 | 115.0 | 687.0  | 4119.0 |   0.6761  | 0.9258 | 0.8738 |
|    12. Mouth    | 1336.0 | 111.0 | 747.0  | 4161.0 |   0.6414  | 0.9233 | 0.8650 |
|     13. Hair    | 1548.0 | 151.0 | 677.0  | 3979.0 |   0.6957  | 0.9111 | 0.8697 |
|     14. Face    | 1481.0 | 121.0 | 679.0  | 4074.0 |   0.6856  | 0.9245 | 0.8741 |
|     15. Eye     | 1956.0 | 200.0 | 977.0  | 3222.0 |   0.6669  | 0.9072 | 0.8148 |
|    16. Torso    | 2269.0 | 200.0 | 1055.0 | 2831.0 |   0.6826  | 0.9190 | 0.8025 |
|     17. Hand    | 1287.0 | 140.0 | 930.0  | 3998.0 |   0.5805  | 0.9019 | 0.8316 |
|     18. Arm     | 1618.0 | 153.0 | 718.0  | 3866.0 |   0.6926  | 0.9136 | 0.8629 |
|     19. Leg     | 1738.0 | 208.0 | 1049.0 | 3360.0 |   0.6236  | 0.8931 | 0.8022 |
|  20. Foot/Shoe  | 1156.0 | 174.0 | 1005.0 | 4020.0 |   0.5349  | 0.8692 | 0.8145 |
|     21. Wing    | 291.0  |  44.0 | 155.0  | 5865.0 |   0.6525  | 0.8687 | 0.9687 |
|  22. Propeller  |  37.0  |  10.0 | 147.0  | 6161.0 |   0.2011  | 0.7872 | 0.9753 |
|  23. Jet engine |  87.0  |  10.0 |  96.0  | 6162.0 |   0.4754  | 0.8969 | 0.9833 |
|    24. Window   | 640.0  |  42.0 | 329.0  | 5344.0 |   0.6605  | 0.9384 | 0.9416 |
|   25. Row Wind  | 212.0  |  12.0 | 316.0  | 5815.0 |   0.4015  | 0.9464 | 0.9484 |
|    26. Wheel    | 671.0  |  44.0 | 463.0  | 5177.0 |   0.5917  | 0.9385 | 0.9202 |
|     27. Door    | 380.0  |  37.0 | 475.0  | 5463.0 |   0.4444  | 0.9113 | 0.9194 |
|  28. Headlight  | 289.0  |  35.0 | 598.0  | 5433.0 |   0.3258  | 0.8920 | 0.9004 |
|  29. Taillight  | 219.0  |  41.0 | 564.0  | 5531.0 |   0.2797  | 0.8423 | 0.9048 |
| 30. Side mirror | 289.0  |  35.0 | 429.0  | 5602.0 |   0.4025  | 0.8920 | 0.9270 |
|   31. Exhaust   | 105.0  |  35.0 | 664.0  | 5551.0 |   0.1365  | 0.7500 | 0.8900 |
|    32. Pedal    |  95.0  |  14.0 | 119.0  | 6127.0 |   0.4439  | 0.8716 | 0.9791 |
|  33. Handlebars | 211.0  |  18.0 | 108.0  | 6018.0 |   0.6614  | 0.9214 | 0.9802 |
|    34. Engine   |  75.0  |  18.0 | 209.0  | 6053.0 |   0.2641  | 0.8065 | 0.9643 |
|     35. Sail    |  46.0  |  3.0  | 115.0  | 6191.0 |   0.2857  | 0.9388 | 0.9814 |
|     36. Mast    |  63.0  |  7.0  | 116.0  | 6169.0 |   0.3520  | 0.9000 | 0.9806 |
|     37. Text    | 196.0  |  17.0 | 450.0  | 5692.0 |   0.3034  | 0.9202 | 0.9265 |
|    38. Label    | 194.0  |  15.0 |  77.0  | 6069.0 |   0.7159  | 0.9282 | 0.9855 |
|  39. Furn. Leg  | 258.0  |  34.0 | 294.0  | 5769.0 |   0.4674  | 0.8836 | 0.9484 |
|  40. Furn. Back | 400.0  |  73.0 | 273.0  | 5609.0 |   0.5944  | 0.8457 | 0.9456 |
|  41. Furn. Seat | 304.0  |  55.0 | 267.0  | 5729.0 |   0.5324  | 0.8468 | 0.9493 |
|  42. Furn. Arm  | 172.0  |  52.0 | 374.0  | 5757.0 |   0.3150  | 0.7679 | 0.9330 |
|     43. Horn    |  14.0  |  5.0  | 112.0  | 6224.0 |   0.1111  | 0.7368 | 0.9816 |
|     44. Rein    |  72.0  |  11.0 | 142.0  | 6130.0 |   0.3364  | 0.8675 | 0.9759 |
|    45. Saddle   |  45.0  |  5.0  | 110.0  | 6195.0 |   0.2903  | 0.9000 | 0.9819 |
|     46. Leaf    | 176.0  |  9.0  |  55.0  | 6115.0 |   0.7619  | 0.9514 | 0.9899 |
|    47. Flower   |  53.0  |  0.0  | 155.0  | 6147.0 |   0.2548  | 1.0000 | 0.9756 |
|  48. Stem/Trunk | 108.0  |  11.0 | 108.0  | 6128.0 |   0.5000  | 0.9076 | 0.9813 |
|     49. Pot     | 170.0  |  9.0  |  57.0  | 6119.0 |   0.7489  | 0.9497 | 0.9896 |
|    50. Screen   | 114.0  |  13.0 |  51.0  | 6177.0 |   0.6909  | 0.8976 | 0.9899 |
|     51. Skin    | 1858.0 | 149.0 | 579.0  | 3769.0 |   0.7624  | 0.9258 | 0.8854 |
|    52. Metal    | 1178.0 | 137.0 | 500.0  | 4540.0 |   0.7020  | 0.8958 | 0.8998 |
|   53. Plastic   | 496.0  |  58.0 | 1090.0 | 4711.0 |   0.3127  | 0.8953 | 0.8194 |
|     54. Wood    | 389.0  |  72.0 | 461.0  | 5433.0 |   0.4576  | 0.8438 | 0.9161 |
|    55. Cloth    | 2197.0 | 278.0 | 515.0  | 3365.0 |   0.8101  | 0.8877 | 0.8752 |
|    56. Furry    | 591.0  |  31.0 | 213.0  | 5520.0 |   0.7351  | 0.9502 | 0.9616 |
|    57. Glass    | 412.0  |  52.0 | 391.0  | 5500.0 |   0.5131  | 0.8879 | 0.9303 |
|   58. Feather   | 197.0  |  60.0 |  59.0  | 6039.0 |   0.7695  | 0.7665 | 0.9813 |
|     59. Wool    |  81.0  |  27.0 | 145.0  | 6102.0 |   0.3584  | 0.7500 | 0.9729 |
|    60. Clear    |  82.0  |  20.0 | 177.0  | 6076.0 |   0.3166  | 0.8039 | 0.9690 |
|    61. Shiny    | 714.0  | 109.0 | 1148.0 | 4384.0 |   0.3835  | 0.8676 | 0.8022 |
|  62. Vegetation | 182.0  |  10.0 |  51.0  | 6112.0 |   0.7811  | 0.9479 | 0.9904 |
|   63. Leather   |  10.0  |  6.0  | 100.0  | 6239.0 |   0.0909  | 0.6250 | 0.9833 |
+-----------------+--------+-------+--------+--------+-----------+--------+--------+
```

## 6. Get the Relation Between Objects by Attribute Representation

```
cd ./other_process
```

First, get the attribute representation of each class, don't forget change the variable `TRAIN_PATH`, `TEST_PATH` and `IMAGE_PATH` in the code:

```
python compute_presentation.py
```

Then, compute the relationship matrix:

```
python compute_distance.py --n-clusters <clusters number> --split split1
```