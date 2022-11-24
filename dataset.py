import os
import cv2
from tqdm import tqdm
from torchvision import transforms
from torch.utils import data

class pokemon_dataset(data.Dataset):
    def __init__(self, path, size = 256):
        super(pokemon_dataset, self).__init__()
        
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
            transforms.CenterCrop(size)
        ])
        
        self.items = []
        for pokemon in tqdm(os.listdir(path)[:], desc = "Loading pokemon type:"):
            pokemon_path = os.path.join(path, pokemon)
            for item in os.listdir(pokemon_path)[:1]:
                file_type = item.split('.')[-1]
                if file_type == "jpg" or file_type == "jpeg" or file_type == "png":
                    item_path = os.path.join(pokemon_path, item)
                    item_image = cv2.imread(item_path, cv2.IMREAD_COLOR)
                    item_image = trans(item_image)
                    self.items.append((pokemon, item_image))
    
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
        pokemon, item = data
        print(pokemon)
        print(item.shape)