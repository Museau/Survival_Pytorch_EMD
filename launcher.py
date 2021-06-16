import argparse
import datetime
import itertools
import os
import pprint
import random
import sys

import dateutil
import dateutil.tz
import numpy as np
import torch

from shutil import copyfile

from trainer import get_algo
from utils.config import cfg, cfg_from_file
from utils.utils import mkdir_p


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    """
    Parser for the command line arguments.

    Returns
    -------
    args : Namespace
        The arguments.
    """
    parser = argparse.ArgumentParser(description="Launch survival experiment.")
    parser.add_argument('--cfg',
                        dest='cfg_file',
                        help="Optional config file.",
                        default="aids3/cox_aids3.yml", type=str)
    parser.add_argument('--gpu',
                        dest='gpu_id',
                        type=str,
                        default='0')
    parser.add_argument('--manual_seed',
                        type=int,
                        help="manual seed",
                        default=1234)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Loading arguments
    args = parse_args()
    args.cfg_file = os.path.join("config", args.cfg_file)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id

    # Timestamp and config load
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    print(f"Timestamp: {timestamp}")
    cfg.TIMESTAMP = timestamp
    output_dir = "results/%s_%s_%s" % (cfg.DATA.DATASET, cfg.CONFIG_NAME, timestamp)
    cfg.OUTPUT_DIR = output_dir
    mkdir_p(output_dir)
    copyfile(args.cfg_file, os.path.join(output_dir, "config.yml"))
    print("Using config:")
    pprint.pprint(cfg)

    # Setting seeds for reproducibility
    print(f"\nPyTorch/Random seed: {args.manual_seed}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manual_seed)

    param_lr = [float(i) for i in cfg.TRAIN.LR]
    param_l2_coeff = [float(i) for i in cfg.TRAIN.L2_COEFF]
    if cfg.TRAIN.MODEL == "emd":
        param_prior = [float(i) for i in cfg.EMD.PRIOR]
        params = list(itertools.product(param_lr, param_l2_coeff, param_prior))
    else:
        params = list(itertools.product(param_lr, param_l2_coeff))

    test_cindices = []
    split_nbrs = eval(cfg.DATA.SPLITS)
    split_nbrs = split_nbrs or [0, 1, 2, 3, 4]
    for split_nb in split_nbrs:

        print(f"\n\n\nSplit: {split_nb}")
        opt_val_cindices = []
        opt_test_cindices = []
        for p in params:

            print(f"\nParam: {p}\n")
            cfg.PARAMS = "LR" + str(p[0]) + "_L2" + str(p[1])
            cfg.TRAIN.LR = p[0]
            cfg.TRAIN.L2_COEFF = p[1]

            if cfg.TRAIN.MODEL == "emd":
                cfg.PARAMS += "_PRIOR" + str(p[2])
                cfg.EMD.PRIOR = p[2]

            algo = get_algo(split_nb)
            score, concat_pred_test = algo.run()

            val_cindex = score['val']['c_index']
            test_cindex = score['test']['c_index']
            print(f"\nVal c_index: {val_cindex}")
            print(f"Test c_index: {test_cindex}")

            opt_val_cindices.append(val_cindex)
            opt_test_cindices.append(test_cindex)

        test_cindex = opt_test_cindices[np.argmax(opt_val_cindices)]
        test_cindices.append(test_cindex)
        print(f"\nSplit test_cindex: {test_cindex}")

    test_cindices = np.array(test_cindices)
    print(f"\n{test_cindices}")
    print(f"\nTest cindex mean, std: {test_cindices.mean()}, {test_cindices.std()}")
