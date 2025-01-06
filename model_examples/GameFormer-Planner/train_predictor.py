# Copyright(c) 2023 Zhiyu Huang
# Copyright 2024 Huawei Technologies Co., Ltd

import os
import stat
import time
import datetime
import logging
import csv
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from GameFormer.predictor import GameFormer
from GameFormer.train_utils import level_k_loss, planning_loss, motion_metrics, initLogging, set_seed, DrivingData

import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch_npu.optim import NpuFusedAdam 


def train_epoch(data_loader, model, optimizer, epoch_idx, profiling_step):
    epoch_loss = []
    epoch_metrics = []
    model.train()

    start_time = time.time()
    total_step = len(data_loader)
    for step, batch in enumerate(data_loader):
        # prepare data
        inputs = {
            'ego_agent_past': batch[0].to(args.device),
            'neighbor_agents_past': batch[1].to(args.device),
            'map_lanes': batch[2].to(args.device),
            'map_crosswalks': batch[3].to(args.device),
            'route_lanes': batch[4].to(args.device)
        }

        ego_future = batch[5].to(args.device)
        neighbors_future = batch[6].to(args.device)
        neighbors_future_valid = torch.ne(neighbors_future[..., :2], 0)

        # call the mdoel
        optimizer.zero_grad()
        level_k_outputs, ego_plan = model(inputs)
        loss, results = level_k_loss(level_k_outputs, ego_future, neighbors_future, neighbors_future_valid)
        prediction = results[:, 1:]
        plan_loss = planning_loss(ego_plan, ego_future)
        loss += plan_loss

        # loss backward
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        # compute metrics
        metrics = motion_metrics(ego_plan, prediction, ego_future, neighbors_future, neighbors_future_valid)
        epoch_metrics.append(metrics)
        epoch_loss.append(loss.item())

        if step > 0 and step % profiling_step == 0:
            if dist.get_rank() == 0:
                now = datetime.datetime.now()
                formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
                avg_train_time = (time.time() - start_time) / profiling_step
                remain_time = avg_train_time * (total_step - step)
                remain_time = str(datetime.timedelta(seconds=int(remain_time)))
                planningADE, planningFDE = epoch_metrics[-1][0], epoch_metrics[-1][1]
                planningAHE, planningFHE = epoch_metrics[-1][2], epoch_metrics[-1][3]
                predictionADE, predictionFDE = epoch_metrics[-1][4], epoch_metrics[-1][5]
                current_lr = optimizer.param_groups[0]['lr']
                epoch_info = f"[{formatted_time}] Epoch [{epoch_idx}] Step [{step}/{total_step}]: avg_train_time: {avg_train_time:.4f}, remain_time: {remain_time}, current_lr: {current_lr}, loss: {loss.item():.4f}, "
                metrics_info = f"planningADE: {planningADE:.4f}, planningFDE: {planningFDE:.4f}, planningAHE: {planningAHE:.4f}, planningFHE: {planningFHE:.4f}, predictionADE: {predictionADE:.4f}, predictionFDE: {predictionFDE:.4f}"
                logging.info(epoch_info + metrics_info)
                start_time = time.time()
                    
    # show metrics
    epoch_metrics = np.array(epoch_metrics)
    planningADE, planningFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    planningAHE, planningFHE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    predictionADE, predictionFDE = np.mean(epoch_metrics[:, 4]), np.mean(epoch_metrics[:, 5])
    epoch_metrics = [planningADE, planningFDE, planningAHE, planningFHE, predictionADE, predictionFDE]
    if dist.get_rank() == 0:
        logging.info("plannerADE: %.4f, plannerFDE: %.4f, plannerAHE: %.4f, plannerFHE: %.4f, predictorADE: %.4f, predictorFDE: %.4f\n", planningADE, planningFDE, planningAHE, planningFHE, predictionADE, predictionFDE)
        epoch_metrics.append(avg_train_time)
    return np.mean(epoch_loss), epoch_metrics


def valid_epoch(data_loader, model):
    epoch_loss = []
    epoch_metrics = []
    model.eval()

    with tqdm(data_loader, desc="Validation", unit="batch") as data_epoch:
        for batch in data_epoch:
           # prepare data
            inputs = {
                'ego_agent_past': batch[0].to(args.device, non_blocking=True),
                'neighbor_agents_past': batch[1].to(args.device, non_blocking=True),
                'map_lanes': batch[2].to(args.device, non_blocking=True),
                'map_crosswalks': batch[3].to(args.device, non_blocking=True),
                'route_lanes': batch[4].to(args.device, non_blocking=True)
            }

            ego_future = batch[5].to(args.device, non_blocking=True)
            neighbors_future = batch[6].to(args.device, non_blocking=True)
            neighbors_future_valid = torch.ne(neighbors_future[..., :2], 0)

            # call the mdoel
            with torch.no_grad():
                level_k_outputs, ego_plan = model(inputs)
                loss, results = level_k_loss(level_k_outputs, ego_future, neighbors_future, neighbors_future_valid)
                prediction = results[:, 1:]
                plan_loss = planning_loss(ego_plan, ego_future)
                loss += plan_loss

            # compute metrics
            metrics = motion_metrics(ego_plan, prediction, ego_future, neighbors_future, neighbors_future_valid)
            epoch_metrics.append(metrics)
            epoch_loss.append(loss.item())
            data_epoch.set_postfix(loss='{:.4f}'.format(epoch_loss[-1]))

    epoch_metrics = np.array(epoch_metrics)
    planningADE, planningFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    planningAHE, planningFHE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    predictionADE, predictionFDE = np.mean(epoch_metrics[:, 4]), np.mean(epoch_metrics[:, 5])
    epoch_metrics = [planningADE, planningFDE, planningAHE, planningFHE, predictionADE, predictionFDE]
    epoch_metrics_tensor = torch.tensor(epoch_metrics, device=model.device).reshape([-1, 1])
    gathered_data = [torch.zeros_like(epoch_metrics_tensor) for _ in range(dist.get_world_size())] 
    dist.all_gather(gathered_data, epoch_metrics_tensor)

    if dist.get_rank() == 0:
        gathered_data = torch.cat(gathered_data, dim=1)
        gathered_data = gathered_data.mean(dim=-1).cpu().numpy().tolist()
        planningADE, planningFDE, planningAHE, planningFHE, predictionADE, predictionFDE = gathered_data
        epoch_metrics = [planningADE, planningFDE, planningAHE, planningFHE, predictionADE, predictionFDE]
        logging.info("val-plannerADE: %.4f, val-plannerFDE: %.4f, val-plannerAHE: %.4f, val-plannerFHE: %.4f, val-predictorADE: %.4f, val-predictorFDE: %.4f\n", planningADE, planningFDE, planningAHE, planningFHE, predictionADE, predictionFDE)

    return np.mean(epoch_loss), epoch_metrics


def model_training(args_, local_rank_):
    # Logging
    log_path = f"./training_log/{args_.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path + 'train.log')

    # ddp setup  
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank_)

    if local_rank_ == 0:
        logging.info("------------- %s -------------", args_.name)
        logging.info("Batch size: %s", args_.batch_size)
        logging.info("Learning rate: %s", args_.learning_rate)
        logging.info("Use device: %s", args_.device)

    # set seed
    set_seed(args_.seed)

    # set up model
    model = GameFormer(encoder_layers=args_.encoder_layers, decoder_levels=args_.decoder_levels, neighbors=args_.num_neighbors)
    if local_rank_ == 0:
        logging.info("Model Params: %d", sum(p.numel() for p in model.parameters()))
    
    device = '{}:{}'.format(args_.device, local_rank_)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank_])
    
    # use NPU fused optimizer
    optimizer = NpuFusedAdam(model.parameters(), lr=args_.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 13, 16, 19, 22, 25, 28], gamma=0.5)

    # training parameters
    train_epochs = args_.train_epochs
    batch_size = args_.batch_size
    profiling_step = args_.profiling_step
    
    # set up data loaders
    train_set = DrivingData(args_.train_set + '/*.npz', args_.num_neighbors)
    valid_set = DrivingData(args_.valid_set + '/*.npz', args_.num_neighbors)
    train_sampler = DistributedSampler(train_set)
    valid_sampler = DistributedSampler(valid_set, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, 
                              sampler=train_sampler, num_workers=args_.workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, pin_memory=True, 
                              sampler=valid_sampler, num_workers=args_.workers)
    if local_rank_ == 0:
        logging.info("Dataset Prepared: %d train data, %d validation data\n", len(train_set), len(valid_set))
    
    # begin training
    for epoch in range(train_epochs):
        if local_rank_ == 0:
            logging.info("Epoch %d/%d", epoch + 1, train_epochs)
        train_loss, train_metrics = train_epoch(train_loader, model, optimizer, epoch, profiling_step)
        val_loss, val_metrics = valid_epoch(valid_loader, model)

        # save to training log
        log = {'epoch': epoch + 1, 'loss': train_loss, 'lr': optimizer.param_groups[0]['lr'], 'val-loss': val_loss, 
               'train-planningADE': train_metrics[0], 'train-planningFDE': train_metrics[1], 
               'train-planningAHE': train_metrics[2], 'train-planningFHE': train_metrics[3], 
               'train-predictionADE': train_metrics[4], 'train-predictionFDE': train_metrics[5],
               'val-planningADE': val_metrics[0], 'val-planningFDE': val_metrics[1], 
               'val-planningAHE': val_metrics[2], 'val-planningFHE': val_metrics[3],
               'val-predictionADE': val_metrics[4], 'val-predictionFDE': val_metrics[5]}

        if dist.get_rank() == 0:
            flags = os.O_RDWR | os.O_CREAT
            mode = stat.S_IWUSR | stat.S_IRUSR
            if epoch == 0:
                with os.fdopen(os.open(f'./training_log/{args_.name}/train_log.csv', flags, mode), 'w') as csv_file:
                    writer = csv.writer(csv_file) 
                    writer.writerow(log.keys())
                    writer.writerow(log.values())
            else:
                with os.fdopen(os.open(f'./training_log/{args_.name}/train_log.csv', flags, mode), 'w') as csv_file:                    
                    writer = csv.writer(csv_file)
                    writer.writerow(log.values())

            # save model at the end of epoch
            torch.save(model.state_dict(), f'training_log/{args_.name}/model_epoch_{epoch+1}_valADE_{val_metrics[0]:.4f}.pth')
            os.chmod(f'training_log/{args_.name}/model_epoch_{epoch+1}_valADE_{val_metrics[0]:.4f}.pth', mode)
            logging.info("Model saved in training_log/%s\n", args_.name)
            
            if epoch == train_epochs - 1:
                logging.info("Model Performance (FPS): %.4f", 1 / train_metrics[-1])
                logging.info("Model Metric (plannerADE): %.4f", val_metrics[0])

        # reduce learning rate
        scheduler.step()

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name (default: "Exp1")', default="Exp1")
    parser.add_argument("--profiling_step", type=int, default=10, help="number of steps for profiling")
    parser.add_argument('--train_set', type=str, help='path to train data')
    parser.add_argument('--valid_set', type=str, help='path to validation data')
    parser.add_argument('--seed', type=int, help='fix random seed', default=1965)
    parser.add_argument("--workers", type=int, default=8, help="number of workers used for dataloader")
    parser.add_argument('--encoder_layers', type=int, help='number of encoding layers', default=3)
    parser.add_argument('--decoder_levels', type=int, help='levels of reasoning', default=2)
    parser.add_argument('--num_neighbors', type=int, help='number of neighbor agents to predict', default=10)
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=30)
    parser.add_argument('--batch_size', type=int, help='batch size (default: 32)', default=32)
    parser.add_argument('--learning_rate', type=float, help='learning rate (default: 1e-4)', default=1e-4)
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='npu')
    args = parser.parse_args()
    local_rank = int(os.environ['LOCAL_RANK'])

    # Run
    model_training(args, local_rank)