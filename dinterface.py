from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from dataset import pokemon_dataset

class DInterface(LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.data_path = args.data_path
        self.image_size = args.image_size
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
    
    def setup(self, stage = None):
        if stage == "fit":
            self.train_dataset = pokemon_dataset(self.data_path, self.image_size)
            self.val_dataset = pokemon_dataset(self.data_path, self.image_size, "test")
        else:
            self.test_dataset = pokemon_dataset(self.data_path, self.image_size, "test")
   
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, 
                          num_workers = self.num_workers, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, 
                          num_workers = self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, 
                          num_workers = self.num_workers, shuffle = False)
        
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, 
                          num_workers = self.num_workers, shuffle = False)