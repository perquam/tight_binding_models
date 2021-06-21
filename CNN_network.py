import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import utils
import os
import numpy as np


import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 9, kernel_size=(5,1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=2)
        self.conv2 = nn.Conv2d(9, 36, kernel_size=(2,1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=2)
        #self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(1440, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)
        x = x.view(-1, 1440)
        x = self.fc1(x)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    
class BandsDataset(Dataset):
    def __init__(self, filename, directory_load, transform=None):
        file_manager = utils.ManagingFiles(directory_save='', directory_load=directory_load)
        self.materials_dataset, self.bands_dataset = file_manager.load_bands_dataset(filename)
        self.transform = transform
        
    def __len__(self):
        return len(self.bands_dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        bands = self.bands_dataset[idx]
        
        mt = self.materials_dataset[idx]
        mt_properties = np.array([mt.Vd2sigma, mt.Vd2pi, mt.Vd2delta, mt.e0, mt.e1, mt.e2]).astype('float')
        
        sample = {'bands': bands, 'material_properties': mt_properties}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
class ToTensor(object):
    def __call__(self, sample):
        bands, mt_properties = sample['bands'], sample['material_properties']
        mt_properties = torch.from_numpy(mt_properties).float()
        bands = np.swapaxes(bands, 0, 1)
        bands = torch.from_numpy(bands.reshape(3, -1, 1)).float()
        #transform = transforms.Compose([transforms.Normalize(mean=[0.4713149820116151, 0.6027473402221133, 0.6972289833654525],
        #                                                      std=[0.14429471, 0.10413322, 0.13522113])])
        #bands = transform(bands)[:3,:,:]
        return {'bands': bands,
                'material_properties': mt_properties}
    
def train(model, device, train_loader, optimizer, epoch_number):
    model.train()
    train_loss = 0.
    for batch_idx, data in enumerate(train_loader):
        data, target = data['bands'].to(device), data['material_properties'].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.L1Loss()
        loss = loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % model_args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.7f}'.format(
                epoch_number, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        train_loss += loss.item()
    train_loss /= len(train_loader)
    print('\nTrain set:      Average loss: {:.7f}'.format(train_loss))
    return train_loss
    

def test(model, device, test_loader, message):
    model.eval()
    test_loss = 0.
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            data, target = data['bands'].to(device), data['material_properties'].to(device)
            output = model(data)
            loss = nn.L1Loss()
            loss = loss(output, target)
            test_loss += loss.item()
            correct += no_correct_batch_preds(output, target)
    test_loss /= len(test_loader)
    print('{}: Average loss: {:.7f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        message, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss

def no_correct_batch_preds(output, target):
    pct = 0.1
    correct_batch = 0
    for i in range(len(output)):
        max_deltas = torch.abs(pct * target[i])
        act_deltas = torch.abs(target[i] - output[i])
        results = act_deltas < max_deltas
        if torch.all(results):
            correct_batch += 1
    return correct_batch
    
def plot_loss(train_loss, validation_loss, title):
    plt.grid(True)
    plt.xlabel("Subsequent epochs")
    plt.ylabel('Average loss')
    plt.plot(range(2, len(train_loss)+1), train_loss[1:], 'o-', label='Training data')
    plt.plot(range(2, len(validation_loss)+1), validation_loss[1:], 'o-', label='Validation data')
    plt.legend()
    plt.title(title)
    
def save_model(model):
    save_dir = os.getcwd() + '/networks/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, 'CNN_model_2')
    save_path = '{}_epoch_{}.pt'.format(save_prefix, model_args['epochs'])
    torch.save(model.state_dict(), save_path)
    
def load_model(model, model_name):
    load_dir = os.getcwd() + '/networks/'
    load_path = load_dir + model_name
    model.load_state_dict(torch.load(load_path))
    return model

if __name__ == '__main__':
    
    plt.rcParams.update({'font.size': 14})

    model_args = {}
    # random seed
    model_args['seed'] = 111
    model_args['batch_size'] = 64
    model_args['lr'] = .001
    model_args['momentum'] = .5
    model_args['epochs'] = 500
    model_args['log_interval'] = 100
    
    
    path_parent = os.path.dirname(os.getcwd())
    
    
    
    training_data_size   = 40000
    validation_data_size = 5000
    test_data_size       = 5000
    
    bands_dataset        = BandsDataset(filename = 'data_normalized_50000_a_const.npy',
                                      directory_load = 'results',
                                      transform=transforms.Compose([ToTensor()]))
    
    
    train_subset, validation_test_subset = torch.utils.data.random_split(bands_dataset, [training_data_size,
                                                                                          validation_data_size + test_data_size])
    validation_subset, test_subset  = torch.utils.data.random_split(validation_test_subset, [validation_data_size,
                                                                                              test_data_size])
    
    loader_kwargs = {'batch_size': model_args['batch_size'], 
                      'num_workers': 2, 
                      'pin_memory': True, 
                      'shuffle': True}
    
    train_loader = torch.utils.data.DataLoader(train_subset, **loader_kwargs)
    validation_loader = torch.utils.data.DataLoader(validation_subset, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_subset, **loader_kwargs)
    
    print(len(train_subset))
    print(len(validation_subset))
    print(len(test_subset))
    print(len(train_loader))
    example_number = 123
    
    device = torch.device('cuda')
    
    model = CNN().to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=model_args['lr'], momentum=model_args['momentum'])
    #optimizer = optim.Adadelta(model.parameters(), lr=model_args['lr'])
    
    torch.manual_seed(model_args['seed'])
    train_loss = []
    validation_loss = []
    for epoch_number in range(1, model_args['epochs'] + 1):
        train_loss.append(train(model, device, train_loader, optimizer, epoch_number))
        validation_loss.append(test(model, device, validation_loader, 'Validation set'))
    
    
    plot1 = plt.figure(1)
    plot_loss(train_loss, validation_loss, 'CNN model')
    
    plt.savefig("CNN model.png")
    plt.show()
    
    save_model(model)
    

