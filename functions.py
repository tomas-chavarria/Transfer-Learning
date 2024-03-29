import torch
import torch.nn as nn


def f_model_instantiation():
    # Opens the pre-trained f model and outlines the number of classes being classified
    f_model = torch.load_state_dict(torch.load('f_50.pth'))
    num_classes = 47

    # Sets the model's layer grad to false, basically "freezing" the layers
    for param in f_model.parameters():
        param.requires_grad = False

    in_features = f_model.fc.in_features

    f_model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    f_model.eval()

    return f_model


# Creates the CNN for image generation, or the 'q'
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 9, kernel_size=1, stride=1, padding=0)
        self.activ1 = nn.Tanh()
        self.conv2 = nn.Conv2d(9, 27, kernel_size=1, stride=1, padding=0)
        self.activ2 = nn.Tanh()
        self.conv3 = nn.Conv2d(27, 9, kernel_size=1, stride=1, padding=0)
        self.activ3 = nn.Tanh()
        self.conv4 = nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0)
        self.activ4 = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activ1(x)
        # print(f"After layer 1: {np.shape(x)}")
        x = self.conv2(x)
        x = self.activ2(x)
        # print(f"After layer 2: {np.shape(x)}")
        x = self.conv3(x)
        x = self.activ3(x)
        # print(f"After layer 3: {np.shape(x)}")
        x = self.conv4(x)
        x = self.activ4(x)
        # print(f"After layer 4: {np.shape(x)}")

        return x
