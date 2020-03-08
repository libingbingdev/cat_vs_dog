import numpy as np
import pandas as pd
import random
import os
import torch
from torch import nn
from torch.autograd import Variable
from PIL import Image
from fractions import Fraction
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import time

#对原压缩文件train中的图片放入Datasets文件夹下
datas = 'Datasets'
#训练集图片，内含cat/dog两个文件夹
train_dir = 'train'
#验证集图片，内含cat/dog两个文件夹
val_dir = 'val'
#测试集图片，内含cat/dog两个文件夹
datatest_dir = 'datatest'
#kaggle测试集图片
test_dir = 'test'

cat_dir = 'cat'
dog_dir = 'dog'

id_label_path = 'sample_submission.csv'
basemodel_result = 'basemodel_submission.csv'
optimalmodel_vgg16 = 'optimalmodel_submission_vgg16.csv'
optimalmodel_densenet121 = 'optimalmodel_submission_densenet121.csv'
optimalmodel_resnet50 = 'optimalmodel_submission_resnet50.csv'
optimalmodel_inception_v3 = 'optimalmodel_submission_inception_v3.csv'
optimalmodel_merge = 'optimalmodel_submission_merge.csv'

TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 32

print_every = 100
basemodel_path = 'basemodel.pth'
vgg16_checkpoint_path = 'vgg16_checkpoint.pth'
densenet121_checkpoint_path = 'densenet121_checkpoint.pth'
resnet50_checkpoint_path = 'resnet50_checkpoint.pth'
inception_v3_checkpoint_path = 'inception_v3_checkpoint.pth'
merge_model_path = 'merge_model.pth'

feature_h5 = 'features.h5'

device = ('cuda' if torch.cuda.is_available() else 'cpu')

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir,transform=train_transforms)
valid_datasets = datasets.ImageFolder(val_dir,transform=valid_transforms)
datatest_datasets = datasets.ImageFolder(datatest_dir,transform=valid_transforms)



# Using the image datasets and the trainforms, define the dataloaders
train_dataloaders = torch.utils.data.DataLoader(train_datasets,batch_size=TRAIN_BATCH_SIZE,shuffle=True)
valid_dataloaders = torch.utils.data.DataLoader(valid_datasets,batch_size=VAL_BATCH_SIZE)
datatest_dataloaders = torch.utils.data.DataLoader(datatest_datasets,batch_size=VAL_BATCH_SIZE)


classes = train_datasets.classes
classes_index = train_datasets.class_to_idx

def dynamic_learning_rate(epoch,parm):
    if epoch < 5:
        return 0.0001*parm
    elif epoch < 10:
        return 0.00005*parm
    elif epoch < 15:
        return 0.00001*parm
    else:
        return 0.000001*parm

def random_samples(path):
    filelist = os.listdir(path)
    indexs = np.random.choice(range(1,len(filelist)),size=10,replace=False)
    samples = [filelist[i] for i in indexs]
    print(samples)
    return samples

def save_checkpoint(epochs,optimizer,model,filepath):
    checkpoint = {'epochs':epochs,
              'optimizer_state_dict':optimizer.state_dict(),
              'model_state_dict':model.state_dict(),
              'classifier':model.classifier,
              'class_to_idx':classes_index}

    torch.save(checkpoint,filepath) 

def fc_save_checkpoint(epochs,optimizer,model,filepath):
    checkpoint = {'epochs':epochs,
              'optimizer_state_dict':optimizer.state_dict(),
              'model_state_dict':model.state_dict(),
              'fc':model.fc,
              'class_to_idx':classes_index}

    torch.save(checkpoint,filepath) 
    
def load_checkpoint(filepath,model_name,device):
    checkpoint = torch.load(filepath,map_location=device)
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier = checkpoint['classifier']
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.classifier = checkpoint['classifier']
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = checkpoint['fc']
    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=True)
        model.aux_logits = False
        model.fc = checkpoint['fc']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def process_image(image,re_size):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model  
    pil_image = Image.open(image)

    width,height = pil_image.size
    #调整图像大小，使最小的边为 256 像素，并保持宽高比
    #使用Fraction(w,h)以分数形式输出宽高比，避免/除法造成的数据失真
    pil_image = pil_image.resize((int(int(re_size*1.14)*(1 if (width <= height) else Fraction(width,height))),
                        int(int(re_size*1.14)*(1 if (height < width) else Fraction(height,width)))),
                        Image.ANTIALIAS)

    # CenterCrop(224):从图像的中心裁剪出 224x224 的部分
    # transforms.ToTensor():把shape=(H x W x C)的像素值范围为[0, 255]的PIL.Image
    #                       转换成shape=(C x H x W)的像素值范围为[0.0, 1.0]的torch.FloatTensor
    # transforms.Normalize将对应的颜色通道进行标准化
    transforms1 = transforms.Compose([transforms.CenterCrop(re_size),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])

    tensor_image = transforms1(pil_image)

    return tensor_image

def one_predict(image_path, re_size,loadmodel, classes_index,topk=2):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    img = process_image(image_path,re_size)
    img = img.unsqueeze(0)
    if torch.cuda.is_available():
        img = img.cuda()
        loadmodel.cuda()
    else:
        img = img.cpu()
        loadmodel.cpu()

    # 将输入变为变量
    img = Variable(img)

    torch.no_grad()
    output = loadmodel(img)
    output = torch.exp(output)
    probs,indexs = output.topk(topk)

    #tensor转为numpy，并降维
    probs = np.squeeze(probs.cpu().detach().numpy())
    indexs = np.squeeze(indexs.cpu().detach().numpy())
    classes = []
    new_dict = {v:k for k,v in classes_index.items()}
    for i in range(len(indexs)):
        classes.append(new_dict.get(indexs[i]))

    return probs,classes

def all_predict(test_dataloader,loadmodel,device,topk=2):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    result_df = pd.DataFrame(columns=['id','label'])
    
    with torch.no_grad():
        loadmodel.to(device)
        loadmodel.eval()
        for ii,(image,image_name) in enumerate(test_dataloader):
            image = image.to(device)
            output = loadmodel(image)
            output = torch.exp(output)
            probs,indexs = output.topk(topk)
            probs = np.squeeze(probs.cpu().detach().numpy()).tolist()
            indexs = np.squeeze(indexs.cpu().detach().numpy()).tolist()
            result_df.loc[result_df.shape[0]] = [int(image_name.numpy()[0]),float(probs[indexs.index(1)])]
    
            if ii%1250==0:
                print('{} pictures have been predicted'.format(ii))
                print('--'*20)
                
    return result_df

def check_logloss_on_test(model,test_dataloaders,criterion,device='cpu'):
    accuracy = 0
    test_loss = 0
    with torch.no_grad():
        model.to(device)
        model.eval()
        for ii,(images,labels) in enumerate(test_dataloaders):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) 
            test_loss += criterion(outputs,labels).item()
            
            ps = torch.exp(outputs).data
            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()
            
    print('Test Loss:{:.3f}...'.format(test_loss/len(test_dataloaders)),
        'Test Accuracy:{:.3f}'.format(accuracy/len(test_dataloaders)))

def plt_result(dataframe):
    plt.plot(dataframe['point'], dataframe['train_loss'], 'bo', label='Training loss')
    plt.plot(dataframe['point'], dataframe['val_loss'], 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

class TestDataset(Dataset):
    
    def __init__(self,id_label_path=None,images_dir=None,transforms=None):
        '''
            id_label_path:存储测试集图片的id和label的csv文件
            images_dir:存储测试集图片的文件夹
            
        '''
        self.id_label_path=id_label_path
        self.images_dir=images_dir
        self.transforms=transforms
        
        self.id_label = pd.read_csv(self.id_label_path)

    #返回整个数据集大小
    def __len__(self):
        return len(self.id_label)
    
    #根据索引index返回dataset[index]
    def __getitem__(self,index):
        #img_id即为图片名称
        img_id,img_label =  self.id_label.iloc[index]
        img_path = os.path.join(self.images_dir,'{}.jpg'.format(int(img_id)))
        image=Image.open(img_path)
        if self.transforms is not None:
            image=self.transforms(image)
        #返回图片以及图片名称(img_id),便于预测完之后存储结果
        return image,img_id

test_datasets = TestDataset(id_label_path,test_dir,transforms=valid_transforms)
test_dataloaders = DataLoader(test_datasets)

def save_result(test_dataloaders,final_loadmodel,device,result_path):
    start = time.time()
    result_df = all_predict(test_dataloaders,final_loadmodel,device)
    end = time.time()
    runing_time = end - start
    print('Test time is {:.0f}m {:.0f}s'.format(runing_time//60,runing_time%60))
    
    result_df['id'] = result_df['id'].astype(int)
    result_df.sort_values(by='id',ascending=True,inplace=True)
    result_df.to_csv(result_path,index=False)

def display_samples(model,re_size):
    samples = np.random.choice(range(1,12500),size=10,replace=False)
    print(samples)
    fig = plt.figure(figsize=(16,6))
    
    for index in range(len(samples)):
        probs, classes = one_predict(os.path.join(test_dir,'{}.jpg'.format(samples[index])),re_size,model,classes_index)
        plt.subplot(2,5,index+1)
        plt.title('{}:{:.3f},{}:{:.3f}'.format(classes[0],probs[0],classes[1],probs[1]))
        img = Image.open(os.path.join(test_dir,'{}.jpg'.format(samples[index])))
        img = img.resize((re_size,re_size))
        plt.imshow(img)
    
    #使得子图横纵坐标更加紧凑，主要用于自动调整图区的大小以及间距，使所有的绘图及其标题、坐标轴标签等都可以不重叠的完整显示在画布上。
    fig.tight_layout()
    plt.show()

def merge_result(df1_path,df2_path,merge_path):
    df1 = pd.pd.read_csv(df1_path)
    df2 = pd.pd.read_csv(df2_path)
    merge_df = df1.copy()
    merge_df['label'] = (merge_df['label'] + df2['label'])/2
    merge_df.to_csv(result_path,index=False)