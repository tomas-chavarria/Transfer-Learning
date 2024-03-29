import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from PIL import Image
import numpy as np
from functions import f_model_instantiation, Generator, Dataset # Dataset not yet created, but will be in the other file

# Set the image directories from the user's input
high_res_directory = "C:\\Users\\themp\\WORK\\Machine Learning and Sensing Lab - Navy Project\\Domain Transfer Neural Net\\X_Pairs"
downsampled_directory = "C:\\Users\\themp\\WORK\\Machine Learning and Sensing Lab - Navy Project\\Domain Transfer Neural Net\\Y_Pairs"

# Creates transforms for the original images in their respective sizes
transform_Y = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

transform_X = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])


# Loss function (use pre-trained discriminator's output as loss)
def generator_loss(fake_outputs):
    return -torch.mean(fake_outputs)


# Split the dataset into training and testing sets
batch_size = 15
input_size = 200
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize the model, loss function, and optimizer
f_model = f_model_instantiation()
q_model = Generator()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(q_model.parameters(), lr=0.001)

# Training loop for image generating CNN
num_epochs = 10
for epoch in range(num_epochs):
    for i in range(len(train_dataloader)):
        # Train Generator
        optimizer.zero_grad()
        noise = torch.randn(batch_size, input_size)
        fake_images = q_model(noise)
        fake_outputs = f_model(fake_images)
        discriminator_loss = criterion(fake_outputs)
        discriminator_loss.backward()
        optimizer.step()

###### IGNORE FOR NOW ######
# Testing loop
# q_model.eval()
# total_rmse = 0.0
# with torch.no_grad():
#     idx = 0
#     for high_res, downsampled in test_dataloader:
#         inputs = Variable(high_res)
#         targets = Variable(downsampled)
#
#         outputs = combined_model(inputs)
#         rmse = criterion(outputs, targets)
#         total_rmse += rmse.item()
#
#         # Save the output images
#         output_image = transforms.ToPILImage()(outputs.squeeze(0).cpu())
#         output_image.save(f'test_output/test_output_{idx}.jpg')
#
#         # Save the original image for comparison
#         real_image = transforms.ToPILImage()(targets.squeeze(0).cpu())
#         real_image.save(f'test_output/test_output_{idx}_real.jpg')
#
#         idx += 1


# Save the trained model
torch.save(q_model.state_dict(), 'q_combined_model.pth')
