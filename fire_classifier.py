import torchvision.models as models
import torch.nn as nn 
import torch.nn.functional as F

model = models.densenet161(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

classifier_input = model.classifier.in_features
num_labels = 2
class FireClassifier(nn.Module):
    def __init__(self):
        super(FireClassifier, self).__init__()
        self.linear1 = nn.Linear(classifier_input, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, num_labels)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.log_softmax(self.linear4(x), dim=1)
        return x

model.classifier = FireClassifier()