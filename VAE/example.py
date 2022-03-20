import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Download MNIST Dataset
train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=False)

# MNist Data Loader
batch_size=50
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# iterator over train set
for batch_idx, (data, _) in enumerate(train_loader):
    print ("In train stage: data size: {}".format(data.size()))
    if batch_idx == 0:
        nelem = data.size(0)
        nrow  = 10
        save_image(data.view(nelem, 1, 28, 28), './images/image_0' + '.png', nrow=nrow)

# iterator over test set
for data, _ in test_loader:
    print ("In test stage: data size: {}".format(data.size()))


# to be finished by you ...
