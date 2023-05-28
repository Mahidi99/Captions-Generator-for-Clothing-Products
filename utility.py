import torch
import torchvision.models as models
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import torch.nn as nn
import numpy as np
import pretrainedmodels
import cv2
import torch.nn.functional as F
import joblib
from ultralytics import YOLO

# custom loss function for multi-head multi-category classification
def loss_fn(outputs, targets):
    o1, o2 = outputs
    t1, t2 = targets
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    return (l1 + l2) / 2

class ClassificationModel():
    
    def __init__(self):
        return
        
    def load(self, model_path, labels_path,  eval=False):
        self.model = torch.load(model_path)
        self.model = nn.Sequential(self.model)
        
        self.labels = open(labels_path, 'r').read().splitlines()
        
        if eval:
            print(model.eval())
        return
    
    def predictAttributes(self, image_path):
        
        device = torch.device("cpu")
        img = Image.open(image_path)
        
        test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                     ])
        
        image_tensor = test_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        inp = Variable(image_tensor)
        inp = inp.to(device)
        output = self.model(inp)
        probabilities = output.data.cpu().numpy()
        attribute_indices = np.argsort(probabilities)[0][::-1]
        predicted_attributes = [self.labels[i] for i in attribute_indices]
        first_two_attributes = predicted_attributes[:2]
        attributes_str = ' '.join(first_two_attributes)
        return attributes_str

        '''print(first_two_attributes)'''
        
        
    def predictCategory(self, image_path):
        
        device = torch.device("cpu")
        img = Image.open(image_path)
        
        test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                     ])
        
        image_tensor = test_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        inp = Variable(image_tensor)
        inp = inp.to(device)
        output = self.model(inp)
        index = output.data.cpu().numpy().argmax()
        return self.labels[index]
        
        

def caption(image_p):
    model_detection = YOLO("C:\\Users\\HP\\Desktop\\Mahidi\\FYP\\My Work\\Fashion Product Caption Generator\\Detect Multiple\\fileW\\weights\\best.pt")

    i=model_detection.predict(source=image_p, conf=0.4)
    
    if len(i[0]) == 0:
        return " "
    else:
        learner = ClassificationModel()
        learner.load("C:\\Users\\HP\\Desktop\\Mahidi\\FYP\\My Work\\Fashion Product Caption Generator\\fashion-ai-main\\fashion-ai-main\\models\\atr-recognition-stage-2-resnet34.pkl", "C:\\Users\\HP\\Desktop\\Mahidi\\FYP\\My Work\\Fashion Product Caption Generator\\fashion-ai-main\\fashion-ai-main\\clothes-categories\\attribute-classes.txt")
        prediction = learner.predictAttributes(image_p)
        learner1 = ClassificationModel()
        learner1.load("C:\\Users\\HP\\Desktop\\Mahidi\\FYP\\My Work\\Fashion Product Caption Generator\\fashion-ai-main\\fashion-ai-main\\cloth_cat_models\\stage-1_resnet34.pkl", "C:\\Users\\HP\\Desktop\\Mahidi\\FYP\\My Work\\Fashion Product Caption Generator\\fashion-ai-main\\fashion-ai-main\\clothes-categories\\classes.txt")
        prediction1 = learner1.predictCategory(image_p)
        return prediction + " " + prediction1.lower()
        '''print(prediction, prediction1)'''
    
    
def captionMulti(image_p):
    learner = ClassificationModel()
    learner.load("C:\\Users\\HP\\Desktop\\Mahidi\\FYP\\My Work\\Fashion Product Caption Generator\\fashion-ai-main\\fashion-ai-main\\multiple-categories\\models\\atr-recognition-stage-2-resnet34.pkl", "C:\\Users\\HP\\Desktop\\Mahidi\\FYP\\My Work\\Fashion Product Caption Generator\\fashion-ai-main\\fashion-ai-main\\multiple-categories\\attribute-classes.txt")
    prediction2 = learner.predictCategory(image_p)
    return prediction2
    

'''//////////////////////////////////////////////////////////////////////////////////////////////////////////////'''


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


def category(image_p2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_detection = YOLO("C:\\Users\\HP\\Desktop\\Mahidi\\FYP\\My Work\\Fashion Product Caption Generator\\Detect Multiple\\fileW\\weights\\best.pt")

    i=model_detection.predict(source=image_p2, conf=0.4)
    
    if len(i[0]) == 0:
        return "No clothing objects detected ! "
    else:
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
        
        # read an image
        image = cv2.imread(image_p2)
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


        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(cv2.imread(image_p2), cv2.COLOR_BGR2RGB)
        return final_labels[0] + " " + final_labels[1].lower()
        print(final_labels[0], final_labels[1])