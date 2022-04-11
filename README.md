# PD-Net
用于作物病虫害识别的网络
1.所需环境：
python==3.7 pytorch==1.7.1

2.权重文件：
病害所需权重文件：https://pan.baidu.com/s/1qeRd9WBE4J2LT-F3r4L6rQ?pwd=axzy 提取码: axzy
虫害所需权重文件：https://pan.baidu.com/s/1eqixWoqzqyfFgk6wzX8AjQ?pwd=i1fw 提取码: i1fw

3.运行步骤：
①将aidemo和ipdemo解压，将下载的病害权重文件放入aidemo下，将虫害权重文件放入ipdemo下。
②验证病害模型性能：
python train_val.py --arch '50' --dataset 'ai' --checkpoints 'aidemo' --valid
验证虫害模型性能：
python train_val.py --arch '50' --dataset 'ip' --checkpoints 'ipdemo' --valid
