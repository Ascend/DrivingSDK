import os
import time
import torch
import torch.nn as nn
import torch_npu
import argparse
import logging
from torch_npu.npu.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from model import pipNet
from data import highwayTrajDataset
from utils_pt import initLogging, maskedNLL, maskedMSE, maskedNLLTest

from torch.utils.data.dataloader import default_collate
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 设置NPU设备
torch.npu.set_device(0)

## Network Arguments
parser = argparse.ArgumentParser(description='Training: Human_like Trajectory Prediction for Autonomous Driving')
# General setting------------------------------------------
parser.add_argument('--use_cuda', action="store_true", help='if use cuda (default: True)', default = True)
parser.add_argument('--use_planning', action="store_false", help='if use planning coupled module (default: True)',default = True)
parser.add_argument('--use_fusion', action="store_false", help='if use targets fusion module (default: True)',default = True)
parser.add_argument('--train_output_flag', action="store_false", help='if concatenate with true maneuver label (default: True)', default = True)
parser.add_argument('--batch_size', type=int, help='batch size to use (default: 64)',  default=64)
parser.add_argument('--learning_rate', type=float, help='learning rate (default: 1e-3)', default=0.001)
parser.add_argument('--tensorboard', action="store_true", help='if use tensorboard (default: True)', default = True)
# IO setting------------------------------------------
parser.add_argument('--grid_size', type=int,  help='default: (25,5)', nargs=2,    default = [25, 5])
parser.add_argument('--in_length', type=int,  help='History sequence (default: 10)',default = 16)    # 1s history traj at 5Hz
parser.add_argument('--out_length', type=int, help='Predict sequence (default: 15)',default = 25)    # 2s future traj at 5Hz
parser.add_argument('--num_lat_classes', type=int, help='Classes of lateral behaviors',     default = 3)
parser.add_argument('--num_lon_classes', type=int, help='Classes of longitute behaviors',   default = 2)
# Network hyperparameters------------------------------------------
parser.add_argument('--temporal_embedding_size', type=int,  help='Embedding size of the input traj', default = 32)
parser.add_argument('--encoder_size', type=int, help='lstm encoder size',  default = 64)
parser.add_argument('--decoder_size', type=int, help='lstm decoder size',  default = 128)
parser.add_argument('--soc_conv_depth', type=int, help='The 1st social conv depth',  default = 64)
parser.add_argument('--soc_conv2_depth', type=int, help='The 2nd social conv depth',  default = 16)
parser.add_argument('--dynamics_encoding_size', type=int,  help='Embedding size of the vehicle dynamic',  default = 32)
parser.add_argument('--social_context_size', type=int,  help='Embedding size of the social context tensor',  default = 80)
parser.add_argument('--fuse_enc_size', type=int,  help='Feature size to be fused',  default = 112)

# 新增Transformer专属参数
parser.add_argument('--num_layers', type=int,  help='Num of Transformer Layers', default = 3)
parser.add_argument('--num_heads', type=int,  help='Number of attention heads', default = 4)
parser.add_argument('--feed_forward_dim', type=int,  help='Dimension of feed forward layer', default = 64)
parser.add_argument('--dropout', type=float,  help='Dropout probability', default = 0.1)
# Training setting------------------------------------------
parser.add_argument('--name', type=str, help='log name (default: "1")', default="npu_train")
parser.add_argument('--train_set', type=str, help='Path to train datasets', default='../autodl-tmp/Train_stop_and_go.mat')
parser.add_argument('--val_set', type=str, help='Path to validation datasets', default='../autodl-tmp/Val_stop_and_go.mat')

parser.add_argument("--num_workers", type=int, default=8, help="number of workers used for dataloader")
parser.add_argument('--pretrain_epochs', type=int, help='epochs of pre-training using MSE', default = 10)
parser.add_argument('--train_epochs',    type=int, help='epochs of training using NLL', default = 20)

# Continue training setting------------------------------------------
parser.add_argument('--start_epoch', type=int, default=None, help='Start epoch for resuming training (optional)')
parser.add_argument('--continue_path', type=str, default="", help="Path to pretrained model checkpoint (optional)")


def train_model():
    args = parser.parse_args()
    ## Logging
    log_path = "./trained_models/{}/".format(args.name)
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+'train.log')
    if args.tensorboard:
        logger = SummaryWriter(log_path + 'train-pre{}-nll{}'.format(args.pretrain_epochs, args.train_epochs))
        logger_val = SummaryWriter(log_path + 'validation-pre{}-nll{}'.format(args.pretrain_epochs, args.train_epochs))
    logging.info("------------- {} -------------".format(args.name))
    logging.info("Batch size : {}".format(args.batch_size))
    logging.info("Learning rate : {}".format(args.learning_rate))
    logging.info("Use Planning Coupled: {}".format(args.use_planning))
    logging.info("Use Target Fusion: {}".format(args.use_fusion))

    ## Initialize network and optimizer
    PiP = pipNet(args)
    PiP = PiP.npu()
        
    start_epoch = 0  # 默认从头开始
    if args.continue_path and os.path.exists(args.continue_path) and args.start_epoch is not None:
        PiP.load_state_dict(torch.load(args.continue_path))
        start_epoch = args.start_epoch
        logging.info(f"Resuming training from epoch {start_epoch}, loaded weights from {args.continue_path}")
    else:
        if not args.continue_path:
            logging.info("No checkpoint path provided.")
        elif not os.path.exists(args.continue_path):
            logging.warning(f"Checkpoint path '{args.continue_path}' does not exist.")
        elif args.start_epoch is None:
            logging.info("Checkpoint path provided but no --start_epoch. Training from scratch.")
        logging.info("Starting training from epoch 0.")

    scaler = GradScaler(init_scale=64.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=1000)


    optimizer = torch.optim.Adam(PiP.parameters(), lr=args.learning_rate)
    crossEnt = nn.BCELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


    ## Initialize training parameters
    pretrainEpochs = args.pretrain_epochs
    trainEpochs    = args.train_epochs
    batch_size     = args.batch_size
    eval_batch_num = 20

    ## Initialize data loaders
    logging.info("Train dataset: {}".format(args.train_set))
    trSet = highwayTrajDataset(path=args.train_set,
                         targ_enc_size=args.social_context_size+args.dynamics_encoding_size,
                         grid_size=args.grid_size,
                         fit_plan_traj=False)
    logging.info("Validation dataset: {}".format(args.val_set))
    valSet = highwayTrajDataset(path=args.val_set,
                          targ_enc_size=args.social_context_size+args.dynamics_encoding_size,
                          grid_size=args.grid_size,
                          fit_plan_traj=True)
    trDataloader =  DataLoader(trSet, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=trSet.collate_fn, prefetch_factor=2, pin_memory=True, persistent_workers=True)
    valDataloader = DataLoader(valSet, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=valSet.collate_fn, prefetch_factor=2, pin_memory=True, persistent_workers=True)
    logging.info("DataSet Prepared : {} train data, {} validation data\n".format(len(trSet), len(valSet)))
    logging.info("Network structure: {}\n".format(PiP))

    ## Training process
    for epoch_num in range(start_epoch, pretrainEpochs + trainEpochs):
        epoch_start_time = time.time()
        epoch_start_time_100 = time.time()
        if epoch_num == 0:
            logging.info('Pretrain with MSE loss')
        elif epoch_num == pretrainEpochs:
            logging.info('Train with NLL loss')
        ## Variables to track training performance:
        avg_time_tr, avg_loss_tr, avg_loss_val = 0, 0, 0
        ## Training status, reclaim after each epoch
        PiP.train()
        PiP.train_output_flag = True
        data_stream = torch.npu.Stream()
        for i, data in enumerate(trDataloader):
            t0 = time.time()
            st_time = time.time()
            nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, targsFut, targsFutMask, lat_enc, lon_enc, _, space_h, dv, v_pre = data
            if args.use_cuda:
                # 在专用流上异步传输
                with torch.npu.stream(data_stream):
                    # 批量异步传输，一次完成
                    nbsHist = nbsHist.contiguous().npu(non_blocking=True)
                    nbsMask = nbsMask.contiguous().npu(non_blocking=True)
                    planFut = planFut.contiguous().npu(non_blocking=True)
                    planMask = planMask.contiguous().npu(non_blocking=True)
                    targsHist = targsHist.contiguous().npu(non_blocking=True)
                    targsEncMask = targsEncMask.contiguous().npu(non_blocking=True)
                    lat_enc = lat_enc.contiguous().npu(non_blocking=True)
                    lon_enc = lon_enc.contiguous().npu(non_blocking=True)
                    targsFut = targsFut.contiguous().npu(non_blocking=True)
                    targsFutMask = targsFutMask.contiguous().npu(non_blocking=True)
                    space_h = space_h.contiguous().npu(non_blocking=True)
                    dv = dv.contiguous().npu(non_blocking=True)
                    v_pre = v_pre.contiguous().npu(non_blocking=True)
                
                # 等待数据传输完成（在计算开始前）
                torch.npu.current_stream().wait_stream(data_stream)
            t1 = time.time()
            # Forward pass
            with autocast():
                fut_pred, lat_pred, lon_pred = PiP(nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, lat_enc, lon_enc, _, 
                                                   space_h, dv, v_pre)
            t2 = time.time()
            if epoch_num < pretrainEpochs:
                # Pre-train with MSE loss to speed up training
                l = maskedMSE(fut_pred, targsFut, targsFutMask)
            else:
                # Train with NLL loss
                l = maskedNLL(fut_pred, targsFut, targsFutMask) + crossEnt(lat_pred, lat_enc) + crossEnt(lon_pred, lon_enc)

            # Back-prop and update weights
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(l).backward()
            torch.nn.utils.clip_grad_norm_(PiP.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()
            t3 = time.time()
            # Track average train loss and average train time:
            batch_time = time.time()-st_time
            avg_loss_tr += l.item()
            avg_time_tr += batch_time

            # For every 100 batches: record loss, validate model, and plot.
            if i % 100 == 99:
                torch.npu.synchronize()
                logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"Epoch {epoch_num+1} | Batch {i+1} | "
                      f"100-batch time: {time.time()-epoch_start_time_100:.2f}s | " 
                      f"Loss: {avg_loss_tr/100:.4f}")
                epoch_start_time_100 = time.time()
                avg_time_tr, avg_loss_tr = 0, 0

        ## Save the model after each epoch______________________________________________________________________________
        avg_loss_val = 0
        with torch.no_grad():
            PiP.eval()
            PiP.train_output_flag = False
            for val_i, data in enumerate(valDataloader):
                nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, targsFut, targsFutMask, lat_enc, lon_enc, idx, space_h, dv, v_pre = data
                if args.use_cuda:
                    nbsHist = nbsHist.npu()
                    nbsMask = nbsMask.npu()
                    planFut = planFut.npu()
                    planMask = planMask.npu()
                    targsHist = targsHist.npu()
                    targsEncMask = targsEncMask.npu()
                    lat_enc = lat_enc.npu()
                    lon_enc = lon_enc.npu()
                    targsFut = targsFut.npu()
                    targsFutMask = targsFutMask.npu()
                    space_h = space_h.npu()
                    dv = dv.npu()
                    v_pre = v_pre.npu()
                if epoch_num < pretrainEpochs:
                    PiP.train_output_flag = True
                    fut_pred, _, _ = PiP(nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask,
                                         lat_enc, lon_enc, idx)
                    l = maskedMSE(fut_pred, targsFut, targsFutMask)
                else:
                    fut_pred, lat_pred, lon_pred = PiP(nbsHist, nbsMask, planFut, planMask, targsHist,
                                        targsEncMask, lat_enc, lon_enc, idx, space_h, dv, v_pre)
                    l = maskedNLLTest(fut_pred, lat_pred, lon_pred, targsFut, targsFutMask, avg_along_time=True)
                avg_loss_val += l.item()
                if val_i == eval_batch_num - 1:
                    break
        # 日志 & tensorboard
        avg_loss_val /= min(eval_batch_num, len(valDataloader))
        logging.info(f"Epoch {epoch_num+1} validation loss: {avg_loss_val:.4f}")
        if args.tensorboard:
            logger_val.add_scalar("RMSE" if epoch_num < pretrainEpochs else "NLL",
                                  avg_loss_val, epoch_num+1)
        epoCount = epoch_num + 1
        if epoCount < pretrainEpochs:
            torch.save(PiP.state_dict(), log_path + "{}-pre{}-nll{}.tar".format(args.name, epoCount, 0))
        else:
            torch.save(PiP.state_dict(), log_path + "{}-pre{}-nll{}.tar".format(args.name, pretrainEpochs, epoCount - pretrainEpochs))
        # 更新学习率（使用验证损失）
        scheduler.step(avg_loss_val)
        if args.tensorboard:
            lr_now = optimizer.param_groups[0]['lr']
            logger.add_scalar("LearningRate", lr_now, epoch_num)
        torch.npu.synchronize()
        logging.info(f"Epoch {epoch_num+1} 耗时: {time.time() - epoch_start_time:.2f} 秒")

    # All epochs finish________________________________________________________________________________________________
    torch.save(PiP.state_dict(), log_path+"{}.tar".format(args.name))
    logging.info("Model saved in trained_models/{}/{}.tar\n".format(args.name, args.name))


if __name__ == '__main__':
    train_model()

