import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

import config
from minterface import MInterface
from dinterface import DInterface

if __name__ == "__main__":
    args = config.load_parser()
    pl.seed_everything(args.seed)
    
    # model = MInterface(args)
    model = MInterface.load_from_checkpoint(args.ckpts_dir, args = args)
    data_model = DInterface(args)
    
    ckpt_cb = ModelCheckpoint(dirpath=f'{args.ckpts_dir}/{args.exp_name}', 
                              filename='{epoch:d}', 
                              save_top_k=-1,
                              every_n_epochs=args.every_n_epochs, 
                              save_on_train_epoch_end = True)
    
    pbar = TQDMProgressBar(refresh_rate=1)
    
    logger = TensorBoardLogger(save_dir = os.path.join(os.getcwd(), "logs"), 
                               name = args.exp_name, 
                               log_graph = False)
    
    trainer = Trainer(callbacks = [ckpt_cb, pbar], 
                      accelerator = args.accelerator, 
                      gpus = args.gpus, 
                      max_epochs = args.max_epochs, 
                      default_root_dir = args.default_root_dir, 
                      logger = logger, 
                      num_sanity_val_steps = args.num_sanity_val_steps, 
                      val_check_interval = args.val_check_interval, 
                      log_every_n_steps = args.log_every_n_steps, 
                      limit_val_batches = args.limit_val_batches)
    
    trainer.fit(model, data_model)
