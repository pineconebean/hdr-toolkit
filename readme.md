# HDR Toolkit

## 训练、测试和评估

（测试和评估理应放在一个步骤，目前有点割裂，后面会考虑进行重构）

- train.py: 模型训练

  ```shell
  # example
  python train.py --model psftd-share --save-dir psftd-share --activation sigmoid --loss l1 --data-path ~/liaojd/data/kalantari-step105/train --val-data-path ~/liaojd/data/kal_val_half --val-interval 1000 --dataset kalantari --ea --epochs 10
  ```

  - `--model`: `adnet`, `ahdrnet`, `psftd-share` （参考文件`hdr_toolkit/networks/__init__.py`中定义的模型）
  - `--save-dir`: 随便输入一个名称即可，不需要输入具体的路径，模型将会保存在`../models/{args.save_dir}`目录中
  - `--activation`: 最终输出图像的卷积层所使用的激活函数，如果需要范围在0到1（如kalantari数据集），就使用`sigmoid`函数，如果需要范围大于0，则使用`relu`
  - `--loss`：可以选择`l1`或`mse`。其中，ADNet和AHDRNet的论文中均使用的是`l1`
  - `--data-path`: 数据的保存路径，最好是输入绝对路径（需要满足一定的保存格式）
  - `--dataset`: 数据集的名称，从`kalantari`和`ntire`中选择
  - `--epochs`: 训练的epoch数目
  - `--batch-size`: 设定训练时使用的batch-size
  - `--use-cpu`：是否使用cpu进行运算（默认会使用`torch.device(cuda:0)`）
  - `--val-data-path`: 验证集数据集路径
  - `--val-interval`: 定义在训练过程中每过多少个batch进行validation，默认为1000
  - `--ea`: 将数据集输入的gamma-corrected的图像改为exposure aligned图像。即对gammga-corrected图像进行开方，具体参考文件`data/data_io.py`中的`ev_align`函数的定义

- test.py: 模型测试

  ```shell
  # example
  python test.py --model-type psftd-share --checkpoint /Users/jundaliao/PycharmProjects/hdr-toolkit/models/psftd-share/ --data kalantari --data-with-gt --input-dir /Users/jundaliao/PycharmProjects/hdr-toolkit/samples --output-dir /Users/jundaliao/PycharmProjects/hdr-toolkit/test_result/ --out-activation sigmoid --ea
  ```

  - `--model-type`: 从`adnet`，`ahdrnet`和`psftd-share`中选择一个（其他模型正在整理中）
  - `--checkpoint`: 保存模型的**目录路径**（models文件夹里面就有）
  - `--data`: 数据集的名称，从`kalantari`和`ntire`中选择
  - `--data-with-gt`: 表明数据集是否有对应的ground truth，如果为`true`且设置了`--write-tonemap-gt`，则会在结果中写入ground truth的tonemap后的结果
  - `--input-dir`: 数据集的输入路径（最好是绝对路径）
  - `--output-dir`: 输出的路径（最好是绝对路径）
  - `--device`: 使用的设备，可从`cpu`或`cuda:0`中选。
  - `--out-activation`: 最终输出图像的卷积层所使用的激活函数，如果需要范围在0到1（如kalantari数据集），就使用`sigmoid`函数，如果需要范围大于0，则使用`relu`
  - `--validation`: 是否同时运行验证过程所保留的模型，默认为不运行
  - `--ea`: 如果训练的时候指定了ea，那么测试的时候同样需要指定ea

- evaluation.py 计算输出图像与对应的ground truth之间的psnr的值，输出的结果会默认保存在`--result-dir`所制定的文件夹下

  ```shell
  # example
  python evaluation.py --result-dir /Users/jundaliao/PycharmProjects/hdr-toolkit/test_result/ --reference-dir /Users/jundaliao/PycharmProjects/hdr-toolkit/samples --dataset kalantari
  ```

  - `--result-dir`: 保存输出结果图像的文件夹路径
  - `--reference-dir`: 保存ground truth的文件夹路径
  - `--dataset`: 数据集的名称，从`kalantari`和`ntire`中选择

## 数据集的准备

### Kalantari

数据集的获取：在 https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/ 获取训练集与数据集。

#### 训练集

训练集的裁剪与增强：

- 命令样例

  ```shell 
  python scripts/kalantari_data_process.py --read kalantari/data/train --write kalantari/data/cropped_train --op crop
  ```

- 输出的目录结构示例

  ```
  ...
  011_0475_long.tif    022_0118_short.tif   032_0594_medium.tif  043_0238_gt.hdr      053_0713_short.tif   064_0357_long.tif
  011_0475_medium.tif  022_0119_gt.hdr      032_0594_short.tif   043_0238_long.tif    053_0714_gt.hdr      064_0357_medium.tif
  ...
  ```

#### 测试集

测试集的重新组织（将所有测试文件重新命名并全部放置在指定的文件夹下）

- 命令样例

  ```shell
  python scripts/kalantari_data_process.py --read kalantari/data/test --write kalantari/data/reorganized_test --op reorganize
  ```

- 输出的目录结构示例

  ```
  001_exposure.txt
  001_gt.hdr
  001_long.tif
  001_medium.tif
  001_short.tif
  ...
  ```

#### 验证集

使用以下命令进行验证集的准备，输出的目录结构和训练集的目录结构一致

```shell
python scripts/kalantari_data_process.py --read kalantari/data/test --write kalantari/data/val --op prepare_val --val-size 6
```

- `--read`: 在之前的训练过程中，采用的策略是从测试集中选择一部分数据来作为验证集（正常来说应该从测试集和验证集不应该有重合？）
- `--write`: 验证集输出的目录路径
- `--op`: `prepare_val` 指定操作为准备验证集
- `--val-size`：验证集所使用的样例的数量，默认为6
- `--include`: 可选参数，可用于指定在选择验证集时包含哪些测试集中的样例，比如`--include 1 13 barbequeday`就会在选择过程中包含测试集中编号为1、13、barbequeday的这三个样例。
- 注意：由于GPU的内存限制，验证集中的样例同样被裁剪为了较小的patch，而这会导致验证结果与用整张图像测试的结果不一致，进一步导致验证最好的结果不一定产生测试最好的结果。

对训练集进行裁剪以及验证集准备完成后，即可使用`train.py`进行在kalantari数据集上的训练，对测试集进行重新组织后，即可使用`test.py`来输出测试结果。

## 环境

- python >= 3.7
- pytorch >= 1.9
- torchvision >= 0.12.0
- opencv-python=4.5.5
- numpy>=1.22.3
- imageio>=2.22.0