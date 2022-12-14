import os
import cv2
import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils import data

class pokemon_dataset(data.Dataset):
    def __init__(self, path, size = 256, split = "train", test_size = 32):
        super(pokemon_dataset, self).__init__()
        
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
            transforms.CenterCrop(size)
        ])
        
        scale = lambda x: (x - 0.5) * 2.0
            
        self.items = []
        
        if split != "train":
            for item in range(test_size):
                self.items.append((torch.randn(1), torch.randn(1)))
            
        else:
            for pokemon_num, pokemon in enumerate(tqdm(os.listdir(path)[:], 
                                             desc = "Loading pokemon type:")):
                pokemon_path = os.path.join(path, pokemon)
                for item in os.listdir(pokemon_path)[:1]:
                    file_type = item.split('.')[-1]
                    if file_type == "jpg" or file_type == "jpeg" or file_type == "png":
                        item_path = os.path.join(pokemon_path, item)
                        item_image = cv2.imread(item_path, cv2.IMREAD_COLOR)
                        item_image = trans(item_image)
                        item_image = scale(item_image)
                        self.items.append((item_image, pokemon_num, pokemon))
    
    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)

if __name__ == "__main__":
    path = "/home/star/Dataset/pokemon"
    dataset = pokemon_dataset(path)
    dataloader = data.DataLoader(dataset, batch_size = 16, shuffle = True)
    
    print(len(dataset))
    print(len(dataloader))
    
    for data in dataloader:
        image, pokemon_num, pokemon = data
        print(pokemon)
        print(pokemon_num)
        print(image.min())
        print(image.max())
        print(image.shape)