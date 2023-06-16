# BPKD : Boundary Privileged Knowledge Distillation For Semantic Segmentation
Authors: Liyang (Akide) Liu, Zihan Wang, Minh Hieu Phan, Bowen Zhang, Yifan Liu*.   *Corresponding author

[[Paper](https://arxiv.org/abs/2306.08075)] [[Github](https://github.com/AkideLiu/BPKD)] [[Docker](https://github.com/orgs/UAws/packages/container/pytorch-sshd/73081261?tag=ngc-pytorch-1.13-mmcv-1.6.0-mmseg-0.26.0-ubuntu-20.04)] [Pretrained models] [Visualization]

---

> **Abstract:** Current approaches for knowledge distillation in semantic segmentation tend to adopt a holistic approach that treats all spatial locations equally. However, for dense prediction tasks, it is crucial to consider the knowledge representation for different spatial locations in a different manner. Furthermore, edge regions between adjacent categories are highly uncertain due to context information leakage, which is particularly pronounced for compact networks. To address this challenge, this paper proposes a novel approach called boundary-privileged knowledge distillation (BPKD). BPKD distills the knowledge of the teacher model's body and edges separately from the compact student model. Specifically, we employ two distinct loss functions: 1) Edge Loss, which aims to distinguish between ambiguous classes at the pixel level in edge regions. 2) Body Loss, which utilizes shape constraints and selectively attends to the inner-semantic regions. Our experiments demonstrate that the proposed BPKD method provides extensive refinements and aggregation for edge and body regions. Additionally, the method achieves state-of-the-art distillation performance for semantic segmentation on three popular benchmark datasets, highlighting its effectiveness and generalization ability. BPKD shows consistent improvements over various lightweight semantic segmentation structures. 
>
> ![image-20230616202628274](https://minio.llycloud.com/image/uPic/image-20230616azcXmW.png)

## Environment

[mmrazor@8b57a07b5e6033dbd0052aeaf0f72668bdaecd00](https://github.com/open-mmlab/mmrazor/commit/8b57a07b5e6033dbd0052aeaf0f72668bdaecd00)

mmseg==0.26.0

mmcv-full==1.6.0

Checkout [requirements.txt](https://github.com/AkideLiu/BPKD_OLD/blob/main/BPKD/requirements.txt) for full requirements

Docker Image :

```
sudo docker run --gpus all -v ~/data:/data \
-e GITHUB_TOKEN="xxxx" \
-e WANDB_TOKEN="xxxx" \
-it --shm-size=64gb ghcr.io/uaws/pytorch-sshd:ngc-pytorch-1.13-mmcv-1.6.0-mmseg-0.26.0-ubuntu-20.04 /bin/bash
```

## Preparing Dataset

1. Cityscapes
2. PASCAL Context
3. ADE20K

According to MMseg: https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md

## Results and models

### Cityscapes

**Table: Performance on Cityscapes Dataset**

| Methods            | FLOPs(G) | Parameters(M) | mIoU(%) | mAcc(%) | Config | Checkpoint |
| ------------------ | -------- | ------------- | ------- | ------- | ------ | ---------- |
| T: PSPNet-R101     | 256.89   | 68.07         | 79.74   | 86.56   |        |            |
| S: PSPNet-R18      | 54.53    | 12.82         | 74.23   | 81.45   |        |            |
| SKDS               | 54.53    | 12.82         | 76.13   | 82.58   |        |            |
| IFVD               | 54.53    | 12.82         | 75.35   | 82.86   |        |            |
| CIRKD              | 54.53    | 12.82         | 76.03   | 82.56   |        |            |
| CWD                | 54.53    | 12.82         | 76.26   | 83.04   |        |            |
| **BPKD(Ours)**     | 54.53    | 12.82         | **77.57** | **84.47** |        |            |
| T: HRNetV2P-W48    | 95.64    | 65.95         | 80.65   | 87.39   |        |            |
| S: HRNetV2P-W18S   | 10.49    | 3.97          | 75.31   | 83.71   |        |            |
| SKDS               | 10.49    | 3.97          | 77.27   | 84.77   |        |            |
| IFVD               | 10.49    | 3.97          | 77.18   | 84.74   |        |            |
| CIRKD              | 10.49    | 3.97          | 77.36   | 84.97   |        |            |
| CWD                | 10.49    | 3.97          | 77.87   | 84.98   |        |            |
| **BPKD(Ours)**     | 10.49    | 3.97          | **78.58** | **85.78** |        |            |
| T: DeeplabV3P-R101 | 255.67   | 62.68         | 80.98   | 88.7    |        |            |
| S: DeeplabV3P+MV2  | 69.60    | 15.35         | 75.29   | 83.11   |        |            |
| SKDS               | 69.60    | 15.35         | 76.05   | 84.14   |        |            |
| IFVD               | 69.60    | 15.35         | 76.97   | 84.85   |        |            |
| CIRKD              | 69.60    | 15.35         | 77.71   | 85.33   |        |            |
| CWD                | 69.60    | 15.35         | 77.97   | 86.68 |        |            |
| **BPKD(Ours)**             | 69.60    | 15.35         | **78.59** | **86.45** |        |            |
| T: ISANet-R101             | 228.21   | 56.80         | 80.61   | 88.29   |        |            |
| S: ISANet-R18              | 54.33    | 12.46         | 73.62   | 80.36   |        |            |
| SKDS                       | 54.33    | 12.46         | 74.99   | 82.61   |        |            |
| IFVD                       | 54.33    | 12.46         | 75.35   | 82.86   |        |            |
| CIRKD                      | 54.33    | 12.46         | 75.41   | 82.92   |        |            |
| CWD                        | 54.33    | 12.46         | 75.43   | 82.64   |        |            |
| **BPKD(Ours)**             | 54.33    | 12.46         | **75.72** | **83.65** |        |            |



### **ADE20K**

| Methods            | FLOPs(G) | Parameters(M) | mIoU(%) | mAcc(%) | Config | Checkpoint |
| ------------------ | -------- | ------------- | ------- | ------- | ------ | ---------- |
| T: PSPNet-R101    | 256\.89  |    68\.07     |    44\.39    |   54\.75   |
| S:PSPnet-R18      |  54\.53  |    12\.82     |    33\.30    |   42\.58   |
| SKDS              |  54\.53  |    12\.82     |    34\.49    |   44\.28   |
| IFVD              |  54\.53  |    12\.82     |    34\.54    |   44\.26   |
| CIRKD             |  54\.53  |    12\.82     |    35\.07    |   45\.38   |
| CWD               |  54\.53  |    12\.82     |    37\.02    |   46\.33   |
| **BPKD(Ours)**    |  54\.53  |    12\.82     |  **38\.51**  | **47\.70** |
| T: HRNetV2P-W48   |  95\.64  |    65\.95     |    42\.02    |   53\.52   |
| S:HRNetV2P-W18S   |  10\.49  |     3\.97     |    31\.38    |   41\.39   |
| SKDS              |  10\.49  |     3\.97     |    32\.57    |   43\.22   |
| IFVD              |  10\.49  |     3\.97     |    32\.66    |   43\.23   |
| CIRKD             |  10\.49  |     3\.97     |    33\.06    |   44\.30   |
| CWD               |  10\.49  |     3\.97     |    34\.00    |   42\.76   |
| **BPKD(Ours)**    |  10\.49  |     3\.97     |  **35\.31**  | **46\.11** |
| T:DeeplabV3P-R101 | 255\.67  |    62\.68     |    45\.47    |   56\.41   |
| S:DeeplabV3P+MV2  |  69\.60  |    15\.35     |    31\.56    |   45\.14   |
| SKDS              |  69\.60  |    15\.35     |    32\.49    |   46\.47   |
| IFVD              |  69\.60  |    15\.35     |    32\.11    |   46\.07   |
| CIRKD             |  69\.60  |    15\.35     |    32\.24    |   46\.09   |
| CWD               |  69\.60  |    15\.35     |    35\.12    |   49\.76   |
| **BPKD(Ours)**    |  69\.60  |    15\.35     |  **35\.49**  | **53\.84** |
| T:ISANet-R101     | 228\.21  |    56\.80     |    43\.80    |   54\.39   |
| S: ISANet-R18     |  54\.33  |    12\.46     |    31\.15    |   41\.21   |
| SKDS              |  54\.33  |    12\.46     |    32\.16    |   41\.80   |
| IFVD              |  54\.33  |    12\.46     |    32\.78    |   42\.61   |
| CIRKD             |  54\.33  |    12\.46     |    32\.82    |   42\.71   |
| CWD               |  54\.33  |    12\.46     |    37\.56    |   45\.79   |
| **BPKD(Ours)**    |  54\.33  |    12\.46     |  **38\.73**  | **47\.92** |



### Pascal Context 59

| Methods            | FLOPs(G) | Parameters(M) | mIoU(%) | mAcc(%) | Config | Checkpoint |
| ------------------ | -------- | ------------- | ------- | ------- | ------ | ---------- |
| T: PSPNet-R101    | 256\.89  |    68\.07     |      52\.47       |   63\.15   |
| S:PSPnet-R18      |  54\.53  |    12\.82     |      43\.79       |   54\.46   |
| SKDS              |  54\.53  |    12\.82     |      45\.08       |   55\.56   |
| IFVD              |  54\.53  |    12\.82     |      45\.97       |   56\.6    |
| CIRKD             |  54\.53  |    12\.82     |      45\.62       |   56\.15   |
| CWD               |  54\.53  |    12\.82     |      45\.99       |   55\.56   |
| **BPKD(Ours)**    |  54\.53  |    12\.82     |    **46\.82**     | **56\.29** |
| T: HRNetV2P-W48   |  95\.64  |    65\.95     |      51\.12       |   61\.39   |
| S:HRNetV2P-W18S   |  10\.49  |     3\.97     |      40\.62       |   51\.43   |
| SKDS              |  10\.49  |     3\.97     |      41\.54       |   52\.18   |
| IFVD              |  10\.49  |     3\.97     |      41\.55       |   52\.24   |
| CIRKD             |  10\.49  |     3\.97     |      42\.02       |   52\.88   |
| CWD               |  10\.49  |     3\.97     |      42\.89       |   53\.37   |
| **BPKD(Ours)**    |  10\.49  |     3\.97     |    **43\.96**     | **54\.51** |
| T:DeeplabV3P-R101 | 255\.67  |    62\.68     |      53\.20       |   64\.04   |
| S:DeeplabV3P+MV2  |  69\.60  |    15\.35     |      41\.01       |   52\.92   |
| SKDS              |  69\.60  |    15\.35     |      42\.07       |   55\.06   |
| IFVD              |  69\.60  |    15\.35     |      41\.73       |   54\.34   |
| CIRKD             |  69\.60  |    15\.35     |      42\.25       |   55\.12   |
| CWD               |  69\.60  |    15\.35     |      43\.74       |   56\.37   |
| **BPKD(Ours)**    |  69\.60  |    15\.35     |    **46\.23**     | **58\.12** |
| T:ISANet-R101     | 228\.21  |    56\.80     |      53\.41       |   64\.04   |
| S: ISANet-R18     |  54\.33  |    12\.46     |      44\.05       |   54\.67   |
| SKDS              |  54\.33  |    12\.46     |      45\.69       |   56\.27   |
| IFVD              |  54\.33  |    12\.46     |      46\.75       |   56\.4    |
| CIRKD             |  54\.33  |    12\.46     |      45\.83       |   56\.11   |
| CWD               |  54\.33  |    12\.46     |      46\.76       |   56\.48   |
| **BPKD(Ours)**    |  54\.33  |    12\.46     |    **47\.25**     | **56\.81** |

## Code

Code Coming Soon ...
