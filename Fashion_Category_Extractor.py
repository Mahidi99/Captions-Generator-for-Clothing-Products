#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install torch torchvision')


# In[2]:


get_ipython().system('pip install pretrainedmodels')


# In[3]:


import os
import torch
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
matplotlib.style.use('ggplot')


# <h2 style="color:black">Function to Clean the Data</h2>

# In[4]:


def clean_data(df):
    """
    this functions removes those rows from the DataFrame for which there are
    no images in the dataset
    """
    drop_indices = []
    print('[INFO]: Checking if all images are present')
    for index, image_id in tqdm(df.iterrows()):
        if not os.path.exists(f"C:\\Users\\HP\\Desktop\\Mahidi\\FYP\\Datasets\\Fashion Product Images\\images\\{image_id.id}.jpg"):
            drop_indices.append(index)
    print(f"[INFO]: Dropping indices: {drop_indices}")
    df.drop(df.index[drop_indices], inplace=True)
    return df


# <h2 style="color:black">Function to Save the Model</h2>

# In[5]:


# save the trained model to disk
def save_model(epochs, model, optimizer, criterion):
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'C:\\Users\\HP\\Desktop\\Mahidi\\FYP\\My Work\\Fashion Product Caption Generator\\Fashion Category Extractor\\category_extraction_model.pth')


# <h2 style="color:black">Function to Save the Loss Plots</h2>

# In[6]:


# save the train and validation loss plots to disk
def save_loss_plot(train_loss, val_loss):
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('C:\\Users\\HP\\Desktop\\Mahidi\\FYP\\My Work\\Fashion Product Caption Generator\\Fashion Category Extractor\\loss.jpg')
    plt.show()


# <h2 style="color:black">Preparing the Label Dictionaries</h2>

# In[7]:


import pandas as pd
import joblib


# <h3 style="color:black">Function to Map the Categories to Numbers and Save Them</h3>

# In[8]:


def save_label_dicts(df):
    # remove rows from the DataFrame which do not have corresponding images
    df = clean_data(df)
    # we will use the 'gender' and 'baseColour' labels
    # mapping 'gender' to numerical values
    cat_list_gender = df['gender'].unique()
    # 5 unique categories for gender
    num_list_gender = {cat:i for i, cat in enumerate(cat_list_gender)}
    # mapping 'baseColour' to numerical values
    cat_list_colour = df['baseColour'].unique()
    # 15 unique categories for 'baseColour'
    num_list_colour = {cat:i for i, cat in enumerate(cat_list_colour)}
    joblib.dump(num_list_gender, 'C:\\Users\\HP\\Desktop\\Mahidi\\FYP\\My Work\\Fashion Product Caption Generator\\Fashion Category Extractor\\num_listGender.pkl')
    joblib.dump(num_list_colour, 'C:\\Users\\HP\\Desktop\\Mahidi\\FYP\\My Work\\Fashion Product Caption Generator\\Fashion Category Extractor\\num_listColour.pkl')
df = pd.read_csv('C:\\Users\\HP\\Desktop\\Mahidi\\FYP\\Datasets\\Fashion Product Images\\styles.csv', usecols=[0, 1, 2, 3, 4, 4, 5, 6, 9])
save_label_dicts(df)


# In[9]:


# # Remove rows with missing values in the 'baseColour' column
# df = df.dropna(subset=['baseColour'])

# # Print the updated dataframe
# print(df.dropna(subset=['baseColour']))
missing_rows = df[df['baseColour'].isna()]
print(missing_rows)


# <h2 style="color:black">Preparing the Dataset</h2>

# In[10]:


pip install opencv-python


# In[11]:


from torch.utils.data import Dataset
import torch
import joblib
import math
import cv2
import torchvision.transforms as transforms


# <h2 style="color:black">Split the DataFrame into Training and Validation Set</h2>

# In[12]:


def train_val_split(df):
    # remove rows from the DataFrame which do not have corresponding images
    df = clean_data(df)
    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    # 90% for training and 10% for validation
    num_train_samples = math.floor(len(df) * 0.90)
    num_val_samples = math.floor(len(df) * 0.10)
    train_df = df[:num_train_samples].reset_index(drop=True)
    val_df = df[-num_val_samples:].reset_index(drop=True)
    return train_df, val_df


# In[13]:


class FashionDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df
        self.num_list_gender = joblib.load('C:\\Users\\HP\\Desktop\\Mahidi\\FYP\\My Work\\Fashion Product Caption Generator\\Fashion Category Extractor\\num_listGender.pkl')
        self.num_list_colour = joblib.load('C:\\Users\\HP\\Desktop\\Mahidi\\FYP\\My Work\\Fashion Product Caption Generator\\Fashion Category Extractor\\num_listColour.pkl')
        
        self.is_train = is_train
        # the training transforms and augmentations
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        # the validation transforms
        if not self.is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        image = cv2.imread(f"C:\\Users\\HP\\Desktop\\Mahidi\\FYP\\Datasets\\Fashion Product Images\\images\\{self.df['id'][index]}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        cat_gender = self.df['gender'][index]
        label_gender = self.num_list_gender[cat_gender]
        cat_colour = self.df['baseColour'][index]
        label_colour = self.num_list_colour[cat_colour]
        # image to float32 tensor
        image = torch.tensor(image, dtype=torch.float32)
        # labels to long tensors
        label_gender = torch.tensor(label_gender, dtype=torch.long)
        label_colour = torch.tensor(label_colour, dtype=torch.long)
        return {
            'image': image,
            'gender': label_gender,
            'colour': label_colour
        }


# <h2 style="color:black">The Loss Function</h2>
# This is a custom function for the loss function. This is because we have three different labels for which we will have three different loss values in each iteration while training. Therefore, we will need to average over the three loss values.

# In[14]:


import torch.nn as nn
# custom loss function for multi-head multi-category classification
def loss_fn(outputs, targets):
    o1, o2 = outputs
    t1, t2 = targets
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    return (l1 + l2) / 2


# <h2 style="color:black">The Deep Learning Model</h2>

# In[15]:


import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels


# In[16]:


class MultiHeadResNet50(nn.Module):
    def __init__(self, pretrained, requires_grad):
        super(MultiHeadResNet50, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained=None)
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
            print('Training intermediate layer parameters...')
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freezing intermediate layer parameters...')
        # change the final layers according to the number of categories
        self.l0 = nn.Linear(2048, 5) # for gender
        self.l1 = nn.Linear(2048, 48) # for baseColour
    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        return l0, l1


# <h2 style="color:black">Training</h2>

# In[17]:


import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


# In[18]:


# define the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# initialize the model
model = MultiHeadResNet50(pretrained=True, requires_grad=False).to(device)
# learning parameters
lr = 0.001
optimizer = optim.Adam(params=model.parameters(), lr=lr)
criterion = loss_fn
batch_size = 32
epochs = 20


# <h2 style="color:black">Preparing the Data Loaders</h2>

# In[19]:


df = pd.read_csv('C:\\Users\\HP\\Desktop\\Mahidi\\FYP\\Datasets\\Fashion Product Images\\styles.csv', usecols=[0, 1, 2, 3, 4, 4, 5, 6, 9])
train_data, val_data = train_val_split(df)
print(f"[INFO]: Number of training sampels: {len(train_data)}")
print(f"[INFO]: Number of validation sampels: {len(val_data)}")
# training and validation dataset
train_dataset = FashionDataset(train_data, is_train=True)
val_dataset = FashionDataset(val_data, is_train=False)
# training and validation data loader
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)


# <h3 style="color:black">The Training Function</h3>

# In[20]:


# training function
def train(model, dataloader, optimizer, loss_fn, dataset, device):
    model.train()
    counter = 0
    train_running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        
        # extract the features and labels
        image = data['image'].to(device)
        gender = data['gender'].to(device)
        colour = data['colour'].to(device)
        
        # zero-out the optimizer gradients
        optimizer.zero_grad()
        
        outputs = model(image)
        targets = (gender, colour)
        loss = loss_fn(outputs, targets)
        train_running_loss += loss.item()
        
        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()
        
    train_loss = train_running_loss / counter
    return train_loss


# <h3 style="color:black">The Validation Function</h3>

# In[21]:


# validation function
def validate(model, dataloader, loss_fn, dataset, device):
    model.eval()
    counter = 0
    val_running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        
        # extract the features and labels
        image = data['image'].to(device)
        gender = data['gender'].to(device)
        colour = data['colour'].to(device)
        
        outputs = model(image)
        targets = (gender, colour)
        loss = loss_fn(outputs, targets)
        val_running_loss += loss.item()
        
    val_loss = val_running_loss / counter
    return val_loss


# <h3 style="color:black">The Training Loop</h3>

# In[22]:


# start the training
train_loss, val_loss = [], []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, train_dataloader, optimizer, loss_fn, train_dataset, device
    )
    val_epoch_loss = validate(
        model, val_dataloader, loss_fn, val_dataset, device
    )
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Validation Loss: {val_epoch_loss:.4f}")

# save the model to disk
save_model(epochs, model, optimizer, criterion)

# save the training and validation loss plot to disk
save_loss_plot(train_loss, val_loss)


# In[25]:


import torch
import cv2
import torchvision.transforms as transforms
import numpy as np
import joblib

# # Import the MultiHeadResNet50 class from the uploaded .ipynb file
# %run '/content/gdrive/MyDrive/Colab Notebooks/Train_Fashion_Category_Extractor.ipynb'
# from Train_Fashion_Category_Extractor import MultiHeadResNet50


# define the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MultiHeadResNet50(pretrained=False, requires_grad=False)
checkpoint = torch.load('C:\\Users\\HP\\Desktop\\Mahidi\\FYP\\My Work\\Fashion Product Caption Generator\\Fashion Category Extractor\\category_extraction_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# In[39]:


# replace with the path to your input image
input_image_path = "C:\\Users\\HP\\Desktop\\Mahidi\\Pictures\\12.jpg"
# read an image
image = cv2.imread(input_image_path)
# keep a copy of the original image for OpenCV functions
orig_image = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# apply image transforms
image = transform(image)
# add batch dimension
image = image.unsqueeze(0).to(device)
# forward pass the image through the model
outputs = model(image)
# extract the two output
output1, output2 = outputs
# get the index positions of the highest label score
out_label_1 = np.argmax(output1.detach().cpu())
out_label_2 = np.argmax(output2.detach().cpu())

import matplotlib.pyplot as plt

# load the label dictionaries
num_list_gender = joblib.load('C:\\Users\\HP\\Desktop\\Mahidi\\FYP\\My Work\\Fashion Product Caption Generator\\Fashion Category Extractor\\num_listGender.pkl')
num_list_colour = joblib.load('C:\\Users\\HP\\Desktop\\Mahidi\\FYP\\My Work\\Fashion Product Caption Generator\\Fashion Category Extractor\\num_listColour.pkl')

# get the keys and values of each label dictionary
gender_keys = list(num_list_gender.keys())
gender_values = list(num_list_gender.values())
colour_keys = list(num_list_colour.keys())
colour_values = list(num_list_colour.values())
final_labels = []

# append the labels by mapping the index position to the values 
final_labels.append(gender_keys[gender_values.index(out_label_1)])
final_labels.append(colour_keys[colour_values.index(out_label_2)])

print(final_labels[0], final_labels[1])


# In[ ]:




