import torch
import torch.nn as nn
import torchvision.transforms as transforms


def Get_Transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
    
        # Adicionado
        transforms.RandomRotation(degrees=18),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),

        # Adicionado agora
        transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.05
            ),

        transforms.RandomAffine(
            degrees=0, 
            translate=(0.15, 0.15), 
            scale=(0.85, 1.15), 
            shear=10),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        # transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)),

        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])


class NeuralNetwork(nn.Module):
    def __init__(self, dropout_rate=.3, num_labels=0):
        super().__init__()

        # Construção das hidden layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))


        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(256, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)

        if num_labels <= 0:
            raise ValueError("It's necessary define a number of labels greater than 0 (num_label:int)")
        else:
            self.fc3 = nn.Linear(128, num_labels)

        # Ativação GeLU
        self.gelu = nn.GELU()
    
    def forward(self, x):
        # Convoluções
        x = self.pool(self.gelu(self.bn1(self.conv1(x))))
        x = self.pool(self.gelu(self.bn2(self.conv2(x))))
        x = self.pool(self.gelu(self.bn3(self.conv3(x))))
        x = self.pool(self.gelu(self.bn4(self.conv4(x))))


        # Pool Global
        x = self.global_pool(x)
        # Flatten
        x = torch.flatten(x, 1)


        # Dropouts
        x = self.dropout1(self.gelu(self.bn_fc1(self.fc1(x))))
        x = self.dropout1(self.gelu(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)

        return x

# Rede neural antiga
# class NeuralNetwork(nn.Module):
#     def __init__(self, dropout_rate=.3, num_labels=0):
#         super().__init__()

#         # Construção das hidden layers
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)

#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)

#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)

#         self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm2d(256)

#         self.pool = nn.MaxPool2d(2, 2)
#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))


#         self.dropout1 = nn.Dropout(dropout_rate)
#         self.dropout2 = nn.Dropout(dropout_rate * 1.2)

#         self.fc1 = nn.Linear(256, 512)
#         self.bn_fc1 = nn.BatchNorm1d(512)
#         self.fc2 = nn.Linear(512, 256)
#         self.bn_fc2 = nn.BatchNorm1d(256)
#         self.fc3 = nn.Linear(256, 128)

#         if num_labels <= 0:
#             raise ValueError("It's necessary define a number of labels greater than 0 (num_label:int)")
#         else:
#             self.fc4 = nn.Linear(128, num_labels) # Saida

#         # Ativação GeLU
#         self.gelu = nn.GELU()
    
#     def forward(self, x):
#         # Convoluções
#         x = self.pool(self.gelu(self.bn1(self.conv1(x))))
#         x = self.pool(self.gelu(self.bn2(self.conv2(x))))
#         x = self.pool(self.gelu(self.bn3(self.conv3(x))))
#         x = self.pool(self.gelu(self.bn4(self.conv4(x))))


#         # Pool Global
#         x = self.global_pool(x)
#         # Flatten
#         x = torch.flatten(x, 1)


#         # Dropouts
#         x = self.dropout1(self.gelu(self.bn_fc1(self.fc1(x))))
#         x = self.dropout1(self.gelu(self.bn_fc2(self.fc2(x))))
#         x = self.dropout2(self.gelu(self.fc3(x)))
#         x = self.fc4(x)

#         return x