import time
import argparse
import logging
import numpy as np
import torch_npu
import torch.npu.amp
from torch.utils.data import DataLoader
from model import pipNet
from data import highwayTrajDataset
from utils_pt import initLogging, maskedMSETest, maskedNLLTest, maskedMAPETest
import os

# 在 model_evaluate 函数内、np.save 调用之前插入
os.makedirs("./results", exist_ok=True)

## Network Arguments
parser = argparse.ArgumentParser(description='Evaluation: Planning-informed Trajectory Prediction for Autonomous Driving')
# General setting------------------------------------------
parser.add_argument('--use_cuda', action='store_false', help='if use cuda (default: True)', default = True)
parser.add_argument('--use_planning', action="store_false", help='if use planning coupled module (default: True)', default = True)
parser.add_argument('--use_fusion', action="store_false", help='if use targets fusion module (default: True)', default = True)
parser.add_argument('--batch_size', type=int, help='batch size to use (default: 64)', default=1)
parser.add_argument('--train_output_flag', action="store_true", help='if concatenate with true maneuver label (default: No)', default = False)
# IO setting------------------------------------------
parser.add_argument('--grid_size', type=int,  help='default: (25,5)', nargs=2,       default = [25, 5])
parser.add_argument('--in_length', type=int,  help='history sequence (default: 16)', default = 16)
parser.add_argument('--out_length', type=int, help='predict sequence (default: 25)', default = 25)
parser.add_argument('--num_lat_classes', type=int, help='Classes of lateral behaviors',   default = 3)
parser.add_argument('--num_lon_classes', type=int, help='Classes of longitute behaviors', default = 2)
# Network hyperparameters------------------------------------------
parser.add_argument('--temporal_embedding_size', type=int,  help='Embedding size of the input traj', default = 32)
parser.add_argument('--encoder_size', type=int, help='lstm encoder size',  default = 64)
parser.add_argument('--decoder_size', type=int, help='lstm decoder size',  default = 128)
parser.add_argument('--soc_conv_depth', type=int, help='The 1st social conv depth',  default = 64)
parser.add_argument('--soc_conv2_depth', type=int, help='The 2nd social conv depth', default = 16)
parser.add_argument('--dynamics_encoding_size', type=int,  help='Embedding size of the vehicle dynamic', default = 32)
parser.add_argument('--social_context_size', type=int,  help='Embedding size of social context tensor',  default = 80)
parser.add_argument('--fuse_enc_size', type=int,  help='Feature size to be fused',   default = 112)
# 新增Transformer专属参数
parser.add_argument('--num_layers', type=int,  help='Num of Transformer Layers', default = 3)
parser.add_argument('--num_heads', type=int,  help='Number of attention heads', default = 4)
parser.add_argument('--feed_forward_dim', type=int,  help='Dimension of feed forward layer', default = 64)
parser.add_argument('--dropout', type=float,  help='Dropout probability', default = 0.1)

## Evaluation setting------------------------------------------

parser.add_argument('--name',     type=str, help='model name', default="npu_train")
parser.add_argument('--test_set', type=str, help='Path to test datasets', default='../datasets/NGSIM/Test_stop_and_go.mat')
parser.add_argument("--num_workers", type=int, default=8, help="number of workers used for dataloader")
parser.add_argument('--metric',   type=str, help='RMSE & NLL is calculated by (agent/sample) based evaluation', default="agent")
parser.add_argument("--plan_info_ds", type=int, default=1, help="N, further downsampling planning information to N*0.2s")

def model_evaluate():

    args = parser.parse_args()
    # 保存所有 batch 的预测、GT、ID
    all_preds_list, all_gts_list, all_ids_list = [], [], []

    ## Initialize network
    PiP = pipNet(args)
    PiP.load_state_dict(torch.load('./trained_models/{}/{}.tar'.format(args.name, args.name)))
    if args.use_cuda:
        PiP = PiP.npu()
        torch.backends.cudnn.benchmark = True
    ## Evaluation Mode
    PiP.eval() 
    PiP.train_output_flag = False
    initLogging(log_file='./trained_models/{}/evaluation.log'.format((args.name).split('-')[0]))

    ## Intialize dataset
    logging.info("Loading test data from {}...".format(args.test_set))
    tsSet = highwayTrajDataset(path=args.test_set,
                               targ_enc_size=args.social_context_size+args.dynamics_encoding_size,
                               grid_size=args.grid_size,
                               fit_plan_traj=True,
                               fit_plan_further_ds=args.plan_info_ds)
    logging.info("TOTAL :: {} test data.".format(len(tsSet)) )
    tsDataloader = DataLoader(tsSet, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=tsSet.collate_fn)

    ## Loss statistic
    logging.info("<{}> evaluated by {}-based MAPE, NLL & RMSE, with planning input of {}s step.".format(args.name, args.metric, args.plan_info_ds*0.2))
    mode1 = 'agent'
    mode2 = 'sample'
    if args.metric == mode1:
        nll_loss_stat = np.zeros((np.max(tsSet.Data[:, 0]).astype(int) + 1,
                                  np.max(tsSet.Data[:, 13:(13 + tsSet.grid_cells)]).astype(int) + 1, args.out_length))
        rmse_loss_stat = np.zeros((np.max(tsSet.Data[:, 0]).astype(int) + 1,
                                   np.max(tsSet.Data[:, 13:(13 + tsSet.grid_cells)]).astype(int) + 1, args.out_length))
        both_count_stat = np.zeros((np.max(tsSet.Data[:, 0]).astype(int) + 1,
                                    np.max(tsSet.Data[:, 13:(13 + tsSet.grid_cells)]).astype(int) + 1, args.out_length))
        mape_loss_stat = np.zeros_like(rmse_loss_stat)

    elif args.metric == mode2:
        rmse_loss = torch.zeros(25).npu()
        rmse_counts = torch.zeros(25).npu()
        nll_loss = torch.zeros(25).npu()
        nll_counts = torch.zeros(25).npu()
    else:
        raise RuntimeError("Wrong type of evaluation metric is specified")
    avg_eva_time = 0
    total_eval_time = 0
    total_eval_samples = 0
    total_eval_batches = 0

    ## Evaluation process
    with torch.no_grad():
        for i, data in enumerate(tsDataloader):
            st_time = time.time()
            nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, targsFut, targsFutMask, lat_enc, lon_enc, idxs, space_h, dv, v_pre = data

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

            # Inference
            with torch.npu.amp.autocast():
                fut_pred, lat_pred, lon_pred = PiP(nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, lat_enc, lon_enc, idxs, space_h, dv, v_pre)

            # Performance metric
            if args.metric == mode1:
                dsIDs, targsIDs = tsSet.batchTargetVehsInfo(idxs)
                l, c = maskedNLLTest(fut_pred, lat_pred, lon_pred, targsFut, targsFutMask, separately=True)
                # Select the trajectory with the largest probability of maneuver label when evaluating by RMSE
                # 初始化为0
                fut_pred_weighted = torch.zeros_like(fut_pred[0])  # [25, B, 5]
                # 获取预测的行为类别
                lat_pred_label = torch.argmax(lat_pred, dim=1).detach().cpu().numpy()  # [B]
                lon_pred_label = torch.argmax(lon_pred, dim=1).detach().cpu().numpy()  # [B]
                # 获取 GT
                lat_gt = lat_enc.argmax(dim=1).detach().cpu().numpy() if lat_enc.shape[1] > 1 else np.zeros_like(lat_pred_label)
                lon_gt = lon_enc.argmax(dim=1).detach().cpu().numpy() if lon_enc.shape[1] > 1 else np.zeros_like(lon_pred_label)
                if i == 0:
                    all_lat_pred, all_lat_gt = [], []
                    all_lon_pred, all_lon_gt = [], []
                all_lat_pred.extend(lat_pred_label.tolist())
                all_lat_gt.extend(lat_gt.tolist())
                all_lon_pred.extend(lon_pred_label.tolist())
                all_lon_gt.extend(lon_gt.tolist())
                # Softmax 处理行为概率
                lat_probs = torch.softmax(lat_pred, dim=1)  # [B, 3]
                lon_probs = torch.softmax(lon_pred, dim=1)  # [B, 2]

                # 遍历所有模式组合（共6种：2 * 3）
                for mode_idx in range(len(fut_pred)):  # mode_idx = 0~5
                    fut = fut_pred[mode_idx]  # [25, B, 5]
                    lon_class = mode_idx // 3
                    lat_class = mode_idx % 3
                    joint_probs = lon_probs[:, lon_class] * lat_probs[:, lat_class]  # [B]
                    
                    # 扩展维度，使 joint_probs 可以广播乘到所有时间步 & 特征
                    joint_probs_expand = joint_probs.unsqueeze(0).unsqueeze(2)  # [1, B, 1]
                    
                    fut_pred_weighted += fut * joint_probs_expand  # 加权求和

                # Using the most probable trajectory

                ll, cc = maskedMSETest(fut_pred_weighted, targsFut, targsFutMask, separately=True)
                mm, cm = maskedMAPETest(fut_pred_weighted, targsFut, targsFutMask, separately=True)
                l = l.detach().cpu().numpy()
                ll = ll.detach().cpu().numpy()
                c = c.detach().cpu().numpy()
                cc = cc.detach().cpu().numpy()
                mm = mm.detach().cpu().numpy()
                cm = cm.detach().cpu().numpy()
                for j, targ in enumerate(targsIDs):
                    dsID = dsIDs[j]
                    nll_loss_stat[dsID, targ, :]   += l[:, j]
                    rmse_loss_stat[dsID, targ, :]  += ll[:, j]
                    both_count_stat[dsID, targ, :]  += c[:, j]
                    mape_loss_stat[dsID, targ, :] += mm[:, j]
            elif args.metric == mode2:
                l, c = maskedNLLTest(fut_pred, lat_pred, lon_pred, targsFut, targsFutMask)
                nll_loss += l.detach()
                nll_counts += c.detach()
                fut_pred_max = fut_pred[0].new_zeros(fut_pred[0].shape)
                for k in range(lat_pred.shape[0]):
                    lat_man = torch.argmax(lat_pred[k, :]).detach()
                    lon_man = torch.argmax(lon_pred[k, :]).detach()
                    indx = lon_man * 3 + lat_man
                    fut_pred_max[:, k, :] = fut_pred[indx][:, k, :]
                l, c = maskedMSETest(fut_pred_max, targsFut, targsFutMask)
                rmse_loss += l.detach()
                rmse_counts += c.detach()

            # Time estimate
            if args.use_cuda:
                torch.npu.synchronize()
            batch_time = time.time() - st_time
            
            num_target_veh = targsFut.shape[1]
            num_nbr_per_target = nbsHist.shape[1]
            num_total_veh = num_target_veh * (1 + num_nbr_per_target)
            
            avg_eva_time += batch_time
            total_eval_time += batch_time
            total_eval_samples += num_total_veh
            total_eval_batches += 1
            
            #logging.info("[Timing] Batch {}: {:.3f} sec | {} targets | {} total vehicles | {:.6f} sec/vehicle".format(
            #    i + 1, batch_time, num_target_veh, num_total_veh, batch_time / num_total_veh))
            # 收集当前 batch 的预测 / GT / 车辆 ID
            batch_pred = fut_pred_weighted[:, :, :2].permute(1, 0, 2).cpu().numpy()  # [B, T, 2]
            batch_gt   = targsFut[:, :, :2].permute(1, 0, 2).cpu().numpy()
            batch_ids  = [[int(ds), int(vid)] for ds, vid in zip(dsIDs, targsIDs)]

            all_preds_list.append(batch_pred)
            all_gts_list.append(batch_gt)
            all_ids_list.extend(batch_ids)


            if i%100 == 99:
                eta = avg_eva_time / 100 * (len(tsSet) / args.batch_size - i)
                logging.info( "Evaluation progress(%):{:.2f}".format( i/(len(tsSet)/args.batch_size) * 100,) +
                              " | ETA(s):{}".format(int(eta)))
                avg_eva_time = 0


    # Result Summary
    if args.metric == mode1:
        # Loss averaged from all predicted vehicles.
        ds_ids, veh_ids = both_count_stat[:,:,0].nonzero()
        num_vehs = len(veh_ids)
        rmse_loss_averaged = np.zeros((args.out_length, num_vehs))
        nll_loss_averaged = np.zeros((args.out_length, num_vehs))
        count_averaged = np.zeros((args.out_length, num_vehs))
        mape_loss_averaged = np.zeros((args.out_length, num_vehs))
        for i in range(num_vehs):
            count_averaged[:, i] = \
                both_count_stat[ds_ids[i], veh_ids[i], :].astype(bool)
            rmse_loss_averaged[:,i] = rmse_loss_stat[ds_ids[i], veh_ids[i], :] \
                                      * count_averaged[:, i] / (both_count_stat[ds_ids[i], veh_ids[i], :] + 1e-9)
            nll_loss_averaged[:,i]  = nll_loss_stat[ds_ids[i], veh_ids[i], :] \
                                      * count_averaged[:, i] / (both_count_stat[ds_ids[i], veh_ids[i], :] + 1e-9)
            mape_loss_averaged[:, i] = mape_loss_stat[ds_ids[i], veh_ids[i], :] \
                                      * count_averaged[:, i] / (both_count_stat[ds_ids[i], veh_ids[i], :] + 1e-9)
        rmse_loss_sum = np.sum(rmse_loss_averaged, axis=1)
        nll_loss_sum = np.sum(nll_loss_averaged, axis=1)
        count_sum = np.sum(count_averaged, axis=1)
        rmseOverall = np.power(rmse_loss_sum / count_sum, 0.5) * 0.3048  # Unit converted from feet to meter.
        nllOverall = nll_loss_sum / count_sum
    elif args.metric == mode2:
        rmseOverall = (torch.pow(rmse_loss / rmse_counts, 0.5) * 0.3048).cpu()
        nllOverall = (nll_loss / nll_counts).cpu()

    # ====== 输出 RMSE 最高的前 10 个车辆 ID ======
    mean_rmse_per_agent = np.mean(rmse_loss_averaged, axis=0)  # shape: [num_veh]
    top_k = 10
    top_err_indices = np.argsort(mean_rmse_per_agent)[-top_k:]
    top_err_ds_ids = [int(ds_ids[i]) for i in top_err_indices]
    top_err_veh_ids = [int(veh_ids[i]) for i in top_err_indices]

    logging.info("Top {} worst RMSE vehicle IDs (dsID, vehID): {}".format(
        top_k, list(zip(top_err_ds_ids, top_err_veh_ids))
    ))
    
    # ====== 输出RMSE 最低的 10 个车辆 ID ======
    bottom_err_indices = np.argsort(mean_rmse_per_agent)[:top_k]
    bottom_err_ds_ids = [int(ds_ids[i]) for i in bottom_err_indices]
    bottom_err_veh_ids = [int(veh_ids[i]) for i in bottom_err_indices]
    
    # 拼接所有 batch
    all_preds = np.concatenate(all_preds_list, axis=0)  # [N, T, 2]
    all_gts   = np.concatenate(all_gts_list, axis=0)
    all_ids   = np.array(all_ids_list)  # [N, 2]

    # 保存为 NPY
    np.save("./results/TestData_pred_stop_and_go.npy", all_preds.transpose(1, 0, 2))  # → [T, N, 2]
    np.save("./results/TestData_true_stop_and_go.npy", all_gts.transpose(1, 0, 2))
    np.save("./results/veh_ids_stop_and_go.npy", all_ids)

    logging.info(" Saved ALL predictions / GT / veh_ids to NPYs")
    
    # 保存每个 vehicle 的 RMSE 时序 + ID
    per_vehicle_rmse = {
        "ds_ids": ds_ids.tolist(),
        "veh_ids": veh_ids.tolist(),
        "rmse_ts": rmse_loss_averaged.tolist()  # shape [25, N]
    }
    np.save("results/per_vehicle_rmse_ts.npy", per_vehicle_rmse)

    # ====== 自动调用行为特征分析 ======
    run_behavior_analysis(
        top10_ids=list(zip(top_err_ds_ids, top_err_veh_ids)),
        bottom10_ids=list(zip(bottom_err_ds_ids, bottom_err_veh_ids)),
        mat_path=args.test_set
    )

    # Print the metrics every 5 time frame (1s)
    logging.info("========== Inference Time Statistics ==========")
    logging.info("Total evaluated vehicles (including neighbors): {}".format(total_eval_samples))
    logging.info("Average inference time per vehicle: {:.10f} seconds".format(total_eval_time / total_eval_samples))
    if total_eval_samples > 0:
        logging.info("Average inference time per sample: {:.10f} seconds".format(total_eval_time / total_eval_samples))
    if total_eval_batches > 0:
        logging.info("Average inference time per batch: {:.10f} seconds".format(total_eval_time / total_eval_batches))
    logging.info("===============================================")

    logging.info("RMSE (m)\t=> {}, Mean={:.3f}".format(rmseOverall[4::5], rmseOverall[4::5].mean()))
    logging.info("NLL (nats)\t=> {}, Mean={:.3f}".format(nllOverall[4::5], nllOverall[4::5].mean()))
    if args.metric == mode1 and 'all_lat_pred' in locals():
        acc_lat = np.mean(np.array(all_lat_pred) == np.array(all_lat_gt))
        acc_lon = np.mean(np.array(all_lon_pred) == np.array(all_lon_gt))
        acc_both = np.mean((np.array(all_lat_pred) == np.array(all_lat_gt)) & (np.array(all_lon_pred) == np.array(all_lon_gt)))
        logging.info(f"Top-1 Accuracy (Lat) = {acc_lat*100:.2f}% | (Lon) = {acc_lon*100:.2f}% | (Joint) = {acc_both*100:.2f}%")
    logging.info("MAPE (%)\t=> {}, Mean={:.2f}%".format(
        np.round(np.mean(mape_loss_averaged[4::5], axis=1) * 100, 2),
        np.mean(mape_loss_averaged[4::5]) * 100
    ))

import h5py

def run_behavior_analysis(top10_ids, bottom10_ids, mat_path):
    LANE_CHANGE_THRESHOLD = 1.0
    HARD_BRAKE_THRESHOLD = 2.5
    ACC_SMOOTHING = 1e-5

    def extract_metrics(track):
        x = track[1]
        y = track[2]
        t = track[0]

        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(t) * 0.1

        vx = dx / (dt + ACC_SMOOTHING)
        vy = dy / (dt + ACC_SMOOTHING)
        speed = np.sqrt(vx ** 2 + vy ** 2)
        acc = np.diff(speed) / (dt[:-1] + ACC_SMOOTHING)

        lane_changes = np.sum(np.abs(np.diff(x)) > LANE_CHANGE_THRESHOLD)
        hard_brakes = np.sum((np.diff(speed) < -HARD_BRAKE_THRESHOLD).astype(int))
        max_acc = np.max(np.abs(acc)) if len(acc) > 0 else 0
        accel_std = np.std(acc) if len(acc) > 0 else 0
        avg_speed_y = np.mean(np.abs(vy))

        return {
            'lane_changes': lane_changes,
            'hard_brakes': hard_brakes,
            'max_accel': max_acc,
            'accel_std': accel_std,
            'avg_speed_y': avg_speed_y
        }

    def analyze_group(group_ids, Tracks):
        all_metrics = []
        for dsID, vehID in group_ids:
            try:
                traj = Tracks[dsID - 1][vehID - 1]
                metrics = extract_metrics(traj)
                all_metrics.append(metrics)
            except Exception as e:
                logging.info(f"Error reading (dsID={dsID}, vehID={vehID}): {e}")
        return all_metrics

    def summarize_metrics(metrics_list):
        if len(metrics_list) == 0:
            return {}
        keys = metrics_list[0].keys()
        summary = {}
        for k in keys:
            values = np.array([m[k] for m in metrics_list])
            summary[k] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'max': np.max(values),
                'min': np.min(values)
            }
        return summary

    def print_summary(title, summary):
        logging.info(f"\n===== {title} =====")
        for k, v in summary.items():
            logging.info(f"{k:15s} | Mean: {v['mean']:.2f} | Std: {v['std']:.2f} | Max: {v['max']:.2f} | Min: {v['min']:.2f}")

    logging.info("\n[行为分析] 加载轨迹数据中...")
    f = h5py.File(mat_path, 'r')
    f_tracks = f['tracks']
    track_cols, track_rows = f_tracks.shape

    Tracks = []
    for i in range(track_rows):
        Tracks.append([np.transpose(f[f_tracks[j][i]][:]) for j in range(track_cols)])

    logging.info("[行为分析] 正在分析 Top10 与 Bottom10...")
    top_metrics = analyze_group(top10_ids, Tracks)
    bottom_metrics = analyze_group(bottom10_ids, Tracks)

    print_summary("Top 10 High-RMSE Vehicles", summarize_metrics(top_metrics))
    print_summary("Bottom 10 Low-RMSE Vehicles", summarize_metrics(bottom_metrics))


if __name__ == '__main__':
    model_evaluate()
