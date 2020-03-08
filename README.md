
# <center>猫狗大战</center>

## 本项目使用的操作系统:

ContOS7.3 

## 机器硬件

阿里云服务器:ecs.gn4-c4g1.xlarge (4 vCPU 30 GiB, GPU计算型 gn4)

镜像:CentOS 7.3(预装NVIDIA GPU驱动和深度学习框架)

## 使用的库

详见附件《environment.yaml》

## 本项目采用的各模型训练时间、模型最终的Vol Loss、模型kaggle score:

(每个单模型训练了15 epoch,采用动态学习率，具体可参考help文件)
    
* **BaseModel(Vgg16)**

    Time:70'2"           

    Vol Loss:0.028
    
    kaggle score:0.05783


* **Densenet121**

    Time:55'4"           

    Vol Loss:0.046
    
    kaggle score:0.06845
    

* **Resnet50**

    Time:54'43"           

    Vol Loss:0.045
    
    kaggle score:0.06008
    
* **Inception v3**

    Time:80'37"                 

    Vol Loss:0.051
    
    kaggle score:0.07339


* **特征向量融合模型**(最优模型)

    (主要耗时在于提取特征向量上，模型训练并未占用多长时间，故在此不做统计)               

    Vol Loss:0.022
    
    kaggle score:0.04775
