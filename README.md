# NN_Fundus
| model                                                        | accuracy | AUC    |
| ------------------------------------------------------------ | -------- | ------ |
| vanilla resnet50                                             | 0.7455   | 0.7385 |
| resnet50 with subtractive normalization                     | 0.7857   | 0.7772 |
| SEnet50 with subtractive normalization                      | 0.7857   | 0.8113 |
| resnet50 with subtractive normalization and higher resolution | 0.7656   | 0.7545 |
| resnet50 with subtractive normalization and data augmentation | 0.7902   | 0.8060 |
| resnet50 with subtractive normalization and stacking (no pretrain) | 0.8348   | 0.8539 |
| SEnet50 with subtractive normalization and stacking (no pretrain) | 0.8203   | 0.8539 |
| vanilla inceptionV3                                          | 0.7054   | 0.7374 |



substractive normalization：降低不同光照等因素带来的色彩差异

data augmentation：各种杂七杂八的数据增强，但其实提升不太明显

higher resolution：resnet正常的输入是224\*224，但因为resnet倒数第二层是一个adaptivePooling，所以其实是可以输入任意大小图片的，所以我试着直接输入448\*448的，但训练会慢几倍，而且效果看起来不是很明显。

stacking：利用双眼信息：把左右眼的各3通道的图片堆叠起来变成6通道，把resnet50的第一层conv的input channel数从3改成6。这里因为改了网络结构就没有pretrain了（也试过正常resnet50不pretrain的，效果其实跟pretrain的差不多，只是容易过拟合而已）

## Mask

* setting：vanilla inceptionv3；100 epoch; test when better_val_auc appears;

* no masking

  | Proportion of blocked area | accuracy | auc    |
  | -------------------------- | -------- | ------ |
  | 0                          | 0.7857   | 0.8043 |

  

* masking the upper and lower parts

  | Proportion of blocked area | accuracy | auc    |
  | -------------------------- | -------- | ------ |
  | 0.8                        | 0.7723   | 0.7850 |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |

  

* masking the outer rim 

  | Proportion of blocked area | accuracy | auc    |
  | -------------------------- | -------- | ------ |
  | 0.8                        | 0.7277   | 0.7536 |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |

* masking a horizontal stripe through the middle

  | Proportion of blocked area | accuracy | auc    |
  | -------------------------- | -------- | ------ |
  | 0.8                        | 0.7098   | 0.7345 |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |

* masking a central core 

  | Proportion of blocked area | accuracy | auc    |
  | -------------------------- | -------- | ------ |
  | 0.8                        | 0.7723   | 0.7412 |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |
  |                            |          |        |

相同比例下mask中间的视盘对acc和auc的影响最大，这与梯度可视化的结果一致。