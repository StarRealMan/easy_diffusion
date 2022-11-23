import os
import configargparse

def load_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, default="config/default.txt")
    
    # User-defined arguments
    parser.add_argument('--exp_name', type = str, default = "test_exp")
    parser.add_argument('--data_path', type = str, default = "/home/star/Dataset/pokemon")
    parser.add_argument('--image_size', type = int, default = 256)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--num_workers', type = int, default = 8)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--lr', type = float, default = 5e-4)
    parser.add_argument('--lr_scheduler', type = str, choices=['step', 'cosine'])
    parser.add_argument('--lr_decay_steps', type = int, default=20)
    parser.add_argument('--lr_decay_rate', type = float, default=0.5)
    parser.add_argument('--lr_decay_min_lr', type = float, default=1e-4)
    parser.add_argument('--ckpts_dir', type=str, default="ckpts/")
    parser.add_argument('--ckpts_epoch', type=int, default=0)
    parser.add_argument('--prediction_dir', type=str, default="predictions/")
    
    # Trainer arguments
    parser.add_argument('--accelerator', type = str, default = "gpu")
    parser.add_argument('--gpus', type = int, default = 1)
    parser.add_argument('--max_epochs', type = int, default = 3000)
    parser.add_argument('--default_root_dir', type = str, default = os.getcwd())
    parser.add_argument('--num_sanity_val_steps', type = int, default = 0)
    parser.add_argument('--val_check_interval', type = int, default = 10)
    parser.add_argument('--log_every_n_steps', type = int, default = 10)
    parser.add_argument('--limit_val_batches', type = int, default = 0)
    parser.add_argument('--every_n_epochs', type = int, default = 300)
    
    args = parser.parse_args()
    args.ckpts_dir = os.path.join(args.ckpts_dir, args.exp_name, 
                                  "epoch=" + str(args.ckpts_epoch) + ".ckpt")
    
    return args
