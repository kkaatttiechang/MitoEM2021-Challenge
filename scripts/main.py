import os, sys
import argparse
import torch
import pdb

sys.path.append(os.getcwd())

from connectomics.config import get_cfg_defaults, save_all_cfg, update_inference_cfg
from connectomics.engine import Trainer
import torch.backends.cudnn as cudnn

def get_args():
    r"""Get args from command lines.
    """
    parser = argparse.ArgumentParser(description="Model Training & Inference")
    parser.add_argument('--config-file', type=str, help='configuration file (yaml)')
    parser.add_argument('--inference', action='store_true', help='inference mode')
    parser.add_argument('--do_h5', action='store_true', help='output h5 directly')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to load the checkpoint')
    # Merge configs from command line (e.g., add 'SYSTEM.NUM_GPUS 8').
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )
    args = parser.parse_args()
    return args

def main():
    r"""Main function.
    """
    # arguments
    args = get_args()

    print("Command line arguments:")
    print(args)


    # configurations
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    try:
        cfg.merge_from_list(args.opts)
    except:
        pass

    if args.inference:
        update_inference_cfg(cfg)

    cfg.freeze()
    print("Configuration details:")
    print(cfg)

    if not os.path.exists(cfg.DATASET.OUTPUT_PATH):
        print('Output directory: ', cfg.DATASET.OUTPUT_PATH)
        os.makedirs(cfg.DATASET.OUTPUT_PATH)
        save_all_cfg(cfg, cfg.DATASET.OUTPUT_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    cudnn.enabled = True
    cudnn.benchmark = True

    mode = 'test' if args.inference else 'train'
    if args.do_h5: 
        print("test validation mode!")

    trainer = Trainer(cfg, device, mode, args.checkpoint)
    if cfg.DATASET.DO_CHUNK_TITLE == 0:
        if args.inference:
            trainer.test(cfg, args.checkpoint[-14:-8])
        else:
            trainer.train()
    else:
        trainer.run_chunk(mode, cfg, args)

if __name__ == "__main__":
    main()
