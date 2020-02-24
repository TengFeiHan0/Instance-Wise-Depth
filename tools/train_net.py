import sys
sys.path.insert(0, './')
from Core.engine.trainer import Trainer
import argparse
import os 
import torch 

from Core.config import cfg
from Core.utils.utils import *
from Core.utils.embedmask_utils import *
from logger import setup_logger


def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
    parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])
    # LOGGING options
    parser.add_argument("--log_frequency",
                                type=int,
                                help="number of batches between each tensorboard log",
                                default=250)
    parser.add_argument("--save_frequency",
                                type=int,
                                help="number of epochs between each save",
                                default=1)
    parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
    parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                                 default="eigen_zhou")
    parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
    parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))
    
    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
        
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)   
    logger = setup_logger("fcos_core", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
        
    trainer = Trainer(cfg,args)
    trainer.train()

if __name__ == "__main__":
    main()    