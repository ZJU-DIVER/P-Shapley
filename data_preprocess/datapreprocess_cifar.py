import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torchvision import models
from sklearn.decomposition import PCA
import torchvision
from torchvision.models import ResNet18_Weights
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from dataeval.options import args_parser
from PIL import Image
def preprocess_func(dataset):
    if dataset == "mnist" or dataset == "fmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: grayscale_to_rgb(x) if x.shape[0] == 1 else x),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset == "cifar-10":
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))
    return transform

def grayscale_to_rgb(image):
    # Convert grayscale image to RGB image
    rgb_image = TF.to_pil_image(image).convert("RGB")
    rgb_tensor = TF.to_tensor(rgb_image)
    return rgb_tensor

trainsize = 500
devsize = 500
testsize = 2000
args = args_parser()
dataset = args.dataset

# Load the appropriate dataset
if dataset == "cifar-10":
    train_dataset = torchvision.datasets.CIFAR10(root='../data/{}/'.format(dataset), train=True, download=False)
    n_components = 32
elif dataset == "fmnist":
    train_dataset = torchvision.datasets.FashionMNIST(root='../data/{}/'.format(dataset), train=True, download=False)
    n_components = 32
elif dataset == "mnist":
    train_dataset = torchvision.datasets.MNIST(root='../data/{}/'.format(dataset), train=True, download=False)
    n_components = 32
else:
    train_dataset, n_components = None, None

# Get the indices of the 'automobile' and 'truck' classes
classes = train_dataset.classes
class_to_idx = train_dataset.class_to_idx
print(class_to_idx)
idxs = [class_to_idx[c] for c in classes if c in ['T-shirt/top', 'Shirt']]

# Filter the dataset to only include 'automobile' and 'truck' data
data = []
targets = []
tmp_targets = np.array(train_dataset.targets, dtype=np.int64)
for idx in idxs:
    data += train_dataset.data[np.where(np.array(train_dataset.targets) == idx)[0]].tolist()
    targets += tmp_targets[np.where(np.array(train_dataset.targets) == idx)[0]].tolist()

# Convert the data and targets to numpy arrays
data = np.array(data)
targets = np.array(targets)

# Shuffle the data and targets in unison
np.random.seed(42)
shuffle_idxs = np.random.permutation(len(data))
data = data[shuffle_idxs]
targets = targets[shuffle_idxs]

# Take the first 4000 samples
data = data[:trainsize+devsize+testsize]
targets = targets[:trainsize+devsize+testsize]
print("Before reduction:", targets[:10], data.shape, targets.shape)
# Extract learned image representations using ResNet-18
# Step 1: Load the pre-trained ResNet-18 model
weights = ResNet18_Weights.DEFAULT
resnet18 = models.resnet18(weights=weights)
resnet18.eval()

# Step 2: Initialize the inference transforms
preprocess = preprocess_func(dataset)

# Step 3: Extract the image representations using ResNet-18
tmp = []
with torch.no_grad():
    for i in range(len(data)):
        img = Image.fromarray(data[i].astype('uint8'))
        # Step 4: Apply inference preprocessing transforms
        batch = preprocess(img).unsqueeze(0)
        # Step 5: Use the model and print the predicted category
        resnet18.fc = nn.Sequential()
        prediction = resnet18(batch).squeeze(0)
        tmp.append(prediction.numpy())
tmp = np.array(tmp)
print("Resnet fc", tmp.shape)

# Step 6: Fit a PCA model and select the first n_components principal components
pca = PCA(n_components=32)
pca.fit(tmp)
print("After PCA", tmp.shape)
# Step 7: Transform the image representations using PCA
image_representations_pca = pca.transform(tmp)

# Use the learned image representations as the new dataX
dataX = image_representations_pca
print(dataX.shape)
dataY = targets
print("After reduction", dataX[:10], dataY[:10], dataX.shape, dataY.shape)
# Split the data into training, development, and testing sets
trnX, tmpX, trnY, tmpY = train_test_split(dataX, dataY, test_size=devsize+testsize, shuffle=True, random_state=42)
devX, tstX, devY, tstY = train_test_split(tmpX, tmpY, test_size=testsize, shuffle=True, random_state=42)


# Take only the first "trainsize" samples for training
trnX = trnX[:trainsize]
trnY = trnY[:trainsize]

# Save the processed data to a file
result = {}
result["trnX"] = trnX
result["trnY"] = trnY
result["devX"] = devX
result["devY"] = devY
result["tstX"] = tstX
result["tstY"] = tstY
print(result["trnX"].shape)

with open("../data/{}_binary.pkl".format(dataset), "wb") as f:
    pickle.dump(result, f)

# Train a logistic regression classifier and evaluate its accuracy
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
clf = LR(solver="liblinear", max_iter=500, random_state=0)
clf.fit(trnX, trnY)
acc = accuracy_score(clf.predict(tstX), tstY)
print(acc)