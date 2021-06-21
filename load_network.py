import torch
from torchvision import datasets, transforms
from CNN_network import BandsDataset
from CNN_network import CNN
from CNN_network import load_model
from CNN_network import ToTensor
from utils import ManagingFiles
from utils import TMDCmaterial
from utils import au

if __name__ == '__main__':
    model = CNN()
    load_model(model, 'CNN_model_.pt')
    model.eval()
    
    model_args = {}
    # random seed
    model_args['seed'] = 111
    model_args['batch_size'] = 64
    model_args['lr'] = .001
    model_args['momentum'] = .5
    model_args['epochs'] = 500
    model_args['log_interval'] = 100
    device = torch.device('cuda')
    test_data_size       = 1000
    
    bands_dataset        = BandsDataset(filename = 'data_10_a_const.npy',
                                      directory_load = 'results',
                                      transform=transforms.Compose([ToTensor()]))
    test_subset, _ = torch.utils.data.random_split(bands_dataset, [10,
                                                                         0])

    loader_kwargs = {'batch_size': model_args['batch_size'], 
                      'num_workers': 2, 
                      'pin_memory': True, 
                      'shuffle': True}
    
    test_loader = torch.utils.data.DataLoader(test_subset, **loader_kwargs)
    
    with torch.no_grad():
        data = next(iter(test_loader))
        data, target = data['bands'].to(device), data['material_properties'].to(device)
        output = model(data)
        output = [TMDCmaterial(0.319, output[0][0] * au.Eh, output[0][1] * au.Eh, output[0][2] * au.Eh, output[0][3] * au.Eh, output[0][4]* au.Eh, output[0][5]* au.Eh)]
        
    file_manager = ManagingFiles(directory_save='results', directory_load='results')
    file_manager.save_bands_dataset(output, target.numpy(), 'MoS2_3_bands_output.npy')