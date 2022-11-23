import os
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import config
from minterface import MInterface
from dinterface import DInterface

if __name__ == "__main__":
    args = config.load_parser()
    pl.seed_everything(args.seed)
    
    model = MInterface.load_from_checkpoint(args.ckpts_dir, args = args)
    data_model = DInterface(args)

    model.eval()
    
    logger = TensorBoardLogger(save_dir = os.path.join(os.getcwd(), "logs"), 
                               name = args.exp_name, 
                               log_graph = False)
    
    trainer = Trainer(accelerator = args.accelerator, 
                      gpus = args.gpus, 
                      max_epochs = args.max_epochs, 
                      default_root_dir = args.default_root_dir, 
                      logger = logger, 
                      val_check_interval = args.val_check_interval, 
                      log_every_n_steps = args.log_every_n_steps)
    
    predictions = trainer.predict(model, data_model)

    predictions = torch.concat(predictions, 0)
    predictions = predictions.permute(0, 2, 3, 1)
    predictions = predictions.cpu().numpy()
    
    if not os.path.exists(args.prediction_dir):
        os.makedirs(args.prediction_dir)
    
    prediction_file = os.path.join(args.prediction_dir, "prediction.npy")
    np.save(prediction_file, predictions)