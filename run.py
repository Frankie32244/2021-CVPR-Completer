# =====================
# COMPLETER: Incomplete Multi-view Clustering via Contrastive Prediction
# =====================
# Author: Yijie Lin
# Date: Mar, 2021
# E-mail: linyijie.gm@gmail.com,

# @inproceedings{lin2021completer,
#    title={COMPLETER: Incomplete Multi-view Clustering via Contrastive Prediction},
#    author={Lin, Yijie and Gou, Yuanbiao and Liu, Zitao and Li, Boyun and Lv, Jiancheng and Peng, Xi},
#    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#    month={June},
#    year={2021}
# }
# =====================

import argparse
import collections
import itertools
import torch

from model import Completer
from get_mask import get_mask
from util import cal_std, get_logger
from datasets import *
from configure import get_default_config


dataset = {
    0: "Caltech101-20",
    1: "Scene_15",
    2: "LandUse_21",
    3: "NoisyMNIST",
}

# 初始化一个参数解析器，内包含四个参数--dataset、--devices、--print_num、--test_time
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='0', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='100', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='5', help='number of test times')

args = parser.parse_args()
dataset = dataset[args.dataset]


def main():
    # Environments 
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # Configure
    config = get_default_config(dataset)
    config['print_num'] = args.print_num
    config['dataset'] = dataset
    logger = get_logger()

    logger.info('Dataset:' + str(dataset))   # print: Dataset:Caltech101-20
    
    for (k, v) in config.items():
        if isinstance(v, dict):
            logger.info("%s={" % (k))
            for (g, z) in v.items():
                logger.info("          %s = %s" % (g, z))     
        else:
            logger.info("%s = %s" % (k, v))
    # --------------------------------------------------------------------------------
    # print:
    # 2024-06-03 04:26:00 - root - INFO: - Prediction={
    # 2024-06-03 04:26:00 - root - INFO: -           arch1 = [128, 256, 128]
    # 2024-06-03 04:26:00 - root - INFO: -           arch2 = [128, 256, 128]
    # 2024-06-03 04:26:00 - root - INFO: - Autoencoder={
    # 2024-06-03 04:26:00 - root - INFO: -           arch1 = [1984, 1024, 1024, 1024, 128]
    # 2024-06-03 04:26:00 - root - INFO: -           arch2 = [512, 1024, 1024, 1024, 128]
    # 2024-06-03 04:26:00 - root - INFO: -           activations1 = relu
    # 2024-06-03 04:26:00 - root - INFO: -           activations2 = relu
    # 2024-06-03 04:26:00 - root - INFO: -           batchnorm = True
    # 2024-06-03 04:26:00 - root - INFO: - training={
    # 2024-06-03 04:26:00 - root - INFO: -           seed = 4
    # 2024-06-03 04:26:00 - root - INFO: -           missing_rate = 0.5
    # 2024-06-03 04:26:00 - root - INFO: -           start_dual_prediction = 100
    # 2024-06-03 04:26:00 - root - INFO: -           batch_size = 256
    # 2024-06-03 04:26:00 - root - INFO: -           epoch = 500
    # 2024-06-03 04:26:00 - root - INFO: -           lr = 0.0001
    # 2024-06-03 04:26:00 - root - INFO: -           alpha = 9
    # 2024-06-03 04:26:00 - root - INFO: -           lambda1 = 0.1
    # 2024-06-03 04:26:00 - root - INFO: -           lambda2 = 0.1
    # 2024-06-03 04:26:00 - root - INFO: - print_num = 100
    # 2024-06-03 04:26:00 - root - INFO: - dataset = Caltech101-20
    # --------------------------------------------------------------------------------
    
    # Load data
    X_list, Y_list = load_data(config)
    x1_train_raw = X_list[0]
    x2_train_raw = X_list[1]

    accumulated_metrics = collections.defaultdict(list)

    for data_seed in range(1, args.test_time + 1):
        # Get the Mask
        np.random.seed(data_seed)
        mask = get_mask(2, x1_train_raw.shape[0], config['training']['missing_rate'])
        # mask the data
        x1_train = x1_train_raw * mask[:, 0][:, np.newaxis]
        x2_train = x2_train_raw * mask[:, 1][:, np.newaxis]

        x1_train = torch.from_numpy(x1_train).float().to(device)
        x2_train = torch.from_numpy(x2_train).float().to(device)
        mask = torch.from_numpy(mask).long().to(device)

        # Set random seeds    设置随机种子目的 : 随机数种子seed确定时，模型的训练结果将始终保持一致。
        if config['training']['missing_rate'] == 0:
            seed = data_seed
        else:
            seed = config['training']['seed']
        np.random.seed(seed)
        random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.backends.cudnn.deterministic = True

        # Build the model  ??? 
        COMPLETER = Completer(config)
        optimizer = torch.optim.Adam(
            itertools.chain(COMPLETER.autoencoder1.parameters(), COMPLETER.autoencoder2.parameters(),
                            COMPLETER.img2txt.parameters(), COMPLETER.txt2img.parameters()),
            lr=config['training']['lr'])
        COMPLETER.to_device(device)

        # Print the models
        logger.info(COMPLETER.autoencoder1)
        # --------------------------------------------------------------------------------
        # Autoencoder1 
        # 2024-06-03 04:26:03 - root - INFO: - Autoencoder(
        #   (_encoder): Sequential(
        #     (0): Linear(in_features=1984, out_features=1024, bias=True)
        #     (1): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (2): ReLU()
        #     (3): Linear(in_features=1024, out_features=1024, bias=True)
        #     (4): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (5): ReLU()
        #     (6): Linear(in_features=1024, out_features=1024, bias=True)
        #     (7): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (8): ReLU()
        #     (9): Linear(in_features=1024, out_features=128, bias=True)
        #     (10): Softmax(dim=1)
        #   )
        #   (_decoder): Sequential(
        #     (0): Linear(in_features=128, out_features=1024, bias=True)
        #     (1): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (2): ReLU()
        #     (3): Linear(in_features=1024, out_features=1024, bias=True)
        #     (4): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (5): ReLU()
        #     (6): Linear(in_features=1024, out_features=1024, bias=True)
        #     (7): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (8): ReLU()
        #     (9): Linear(in_features=1024, out_features=1984, bias=True)
        #     (10): BatchNorm1d(1984, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (11): ReLU()
        #   )
        # )
        # --------------------------------------------------------------------------------

        
        logger.info(COMPLETER.img2txt)
        # --------------------------------------------------------------------------------
        #          2024-06-03 04:26:03 - root - INFO: - Prediction(
        #   (_encoder): Sequential(
        #     (0): Linear(in_features=128, out_features=128, bias=True)
        #     (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (2): ReLU()
        #     (3): Linear(in_features=128, out_features=256, bias=True)
        #     (4): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (5): ReLU()
        #     (6): Linear(in_features=256, out_features=128, bias=True)
        #     (7): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (8): ReLU()
        #   )
        #   (_decoder): Sequential(
        #     (0): Linear(in_features=128, out_features=256, bias=True)
        #     (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (2): ReLU()
        #     (3): Linear(in_features=256, out_features=128, bias=True)
        #     (4): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (5): ReLU()
        #     (6): Linear(in_features=128, out_features=128, bias=True)
        #     (7): Softmax(dim=1)
        #   )
        # )
        # --------------------------------------------------------------------------------
        
        logger.info(optimizer)
        # --------------------------------------------------------------------------------
        #         2024-06-03 04:26:03 - root - INFO: - Adam (
        # Parameter Group 0
        #     amsgrad: False
        #     betas: (0.9, 0.999)
        #     capturable: False
        #     differentiable: False
        #     eps: 1e-08
        #     foreach: None
        #     fused: None
        #     lr: 0.0001
        #     maximize: False
        #     weight_decay: 0
        # )
        # --------------------------------------------------------------------------------
        
        
        # Training
        acc, nmi, ari = COMPLETER.train(config, logger, x1_train, x2_train, Y_list,
                                        mask, optimizer, device)
        accumulated_metrics['acc'].append(acc)
        accumulated_metrics['nmi'].append(nmi)
        accumulated_metrics['ari'].append(ari)

       logger.info('--------------------Training over--------------------')
       cal_std(logger, accumulated_metrics['acc'], accumulated_metrics['nmi'], accumulated_metrics['ari'])


if __name__ == '__main__':
    main()
