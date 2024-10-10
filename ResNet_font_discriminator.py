import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import json
import os
from PIL import Image
import random
import ssl


ssl._create_default_https_context = ssl._create_unverified_context




class CustomImageDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        with open(json_file, 'r') as f:
            self.labels = json.load(f)
        self.root_dir = root_dir
        self.transform = transform

        self.font_types = list(set([v['폰트종류'] for v in self.labels.values()]))  # 폰트 종류의 리스트
        self.font_weights = list(set([v['폰트굵기'] for v in self.labels.values()]))  # 폰트 굵기의 리스트
        self.font_colors = list(set([tuple(v['폰트색상']) for v in self.labels.values()])) 


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, list(self.labels.keys())[idx])
        image = Image.open(img_name)
        label = self.labels[list(self.labels.keys())[idx]]

        # 레이블을 인덱스 값으로 변환
        font_type_idx = self.font_types.index(label['폰트종류'])
        font_weight_idx = self.font_weights.index(label['폰트굵기'])
        font_color = torch.tensor(label['폰트색상'], dtype=torch.float32) / 255.0

        if self.transform:
            image = self.transform(image)

        return image, (font_type_idx, font_weight_idx, font_color)


class MyModel(nn.Module):
    def __init__(self, num_font_types, num_font_weights):
        super(MyModel, self).__init__()
        self.base_model = models.segmentation.fcn_resnet50(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Add an average pooling layer
        self.fc1 = nn.Linear(21, num_font_types)  # Adjust the input size for fc1 and fc2
        self.fc2 = nn.Linear(21, num_font_weights)
        self.fc3 = nn.Linear(21, 3)

    def forward(self, x):
        x = self.base_model(x)['out']  # The output of the base model is the input of our new layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out3 = self.fc3(x)
        return out1, out2, out3

if __name__ == '__main__':
    
    data_transforms = {
    'train': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([# 수정됨
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}




    dataset = CustomImageDataset(json_file='labels.json', root_dir='images', transform=data_transforms['train'])

    train_ratio = 0.8 
    train_size = int(train_ratio * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)




    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_font_types = len(dataset.font_types)
    num_font_weights = len(dataset.font_weights)

    model = MyModel(num_font_types, num_font_weights)
    model.to(device)

    # 모든 파라미터를 최적화
    optimizer1 = optim.Adam([{'params': model.fc1.parameters()}], lr=0.001)
    optimizer2 = optim.Adam([{'params': model.fc2.parameters()}], lr=0.001)
    optimizer3 = optim.Adam([{'params': model.fc3.parameters()}], lr=0.001)
    exp_lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=7, gamma=0.1)
    exp_lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=7, gamma=0.1)
    exp_lr_scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=7, gamma=0.1)

    # 7 에폭마다 0.1씩 학습률 감소
    criterion = nn.CrossEntropyLoss()
    criterion_color = nn.MSELoss()  # use Mean Squared Error for the color prediction

    num_epochs = 30
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = [label.to(device) for label in labels]

                optimizer1.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs1, outputs2, outputs3 = model(inputs)
                    loss1 = criterion(outputs1, labels[0])
                    loss2 = criterion(outputs2, labels[1])

                    labels_color_normalized = labels[2].float() / 255  
                    loss3 = criterion_color(outputs3, labels_color_normalized)

                    # Compute the number of correct predictions
                    _, preds1 = torch.max(outputs1, 1)
                    _, preds2 = torch.max(outputs2, 1)
                    
                    running_corrects += torch.sum(preds1 == labels[0]) + torch.sum(preds2 == labels[1])

                    if phase == 'train':
                        loss1.backward(retain_graph=True) 
                        optimizer1.step()

                        loss2.backward(retain_graph=True)
                        optimizer2.step()

                        loss3.backward()
                        optimizer3.step()
                    running_loss += loss1.item() + loss2.item() + loss3.item()
                    print('0')

            if phase == 'train':
                exp_lr_scheduler1.step()
                exp_lr_scheduler2.step()
                exp_lr_scheduler3.step() 
            
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / (dataset_sizes[phase] * 2)  # 2 for two outputs

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        print()
        torch.save(model.state_dict(), 'Font_Discriminator.pth')

    print('Training complete')
