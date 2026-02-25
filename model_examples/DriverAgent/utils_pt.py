import math
import logging
import numpy as np
import torch
from scipy.optimize import curve_fit

## Network parameters initialization
def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.1)


def initLogging(log_file: str, level: str = "INFO"):
    logging.basicConfig(filename=log_file, filemode='a',
                        level=getattr(logging, level, None),
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())


## Quintic spline definition.
# pylint: disable=too-many-arguments,huawei-too-many-arguments
def quintic_spline(x, z, a, b, c, d, e):
    return z + a * x + b * x ** 2 + c * x ** 3 + d * x ** 4 + e * x ** 5


## Fitting the trajectory of one planning circle by quintic spline, with the current location fixed.
def fitting_traj_by_qs(x, y):
    param, loss = curve_fit(quintic_spline, x, y,
        bounds=([y[0], -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], [y[0]+1e-6, np.inf, np.inf, np.inf, np.inf, np.inf]))
    return param


## Custom activation for output layer (Graves, 2015)
def outputActivation(x, displacement=True):
    if displacement:
        # Then mu value denotes displacement.
        x[:, :, 0:2] = torch.stack([torch.sum(x[0:i, :, 0:2], dim=0) for i in range(1, x.shape[0] + 1)], 0)
    # Each output has 5 params to describe the gaussian distribution.
    muX = x[:, :, 0:1]
    muY = x[:, :, 1:2]
    sigX = x[:, :, 2:3]

    sigY = x[:, :, 3:4]
    rho = x[:, :, 4:5]
    sigX = torch.exp(sigX)  # This positive value represents Reciprocal of SIGMA (1/sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)   # -1 < rho < 1
    out = torch.cat([muX, muY, sigX, sigY, rho], dim=2)
    return out


def maskedNLL(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    sigX = y_pred[:, :, 2]
    sigY = y_pred[:, :, 3]
    rho = y_pred[:, :, 4]
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = 0.5 * torch.pow(ohr, 2) * \
        (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho *
        torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) \
        + torch.log(torch.tensor(2 * math.pi))
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc) / torch.sum(mask)

    return lossVal


# pylint: disable=too-many-arguments,huawei-too-many-arguments
def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,
                  num_lat_classes=3, num_lon_classes=2,
                  use_maneuvers=True, avg_along_time=False, separately=False):
    """
    推理 / 验证阶段的 NLL 计算
    现在的 sigX、sigY 直接就是 precision (= 1/σ)，不再二次 exp
    """

    # 保证正数以防 log/除零
    def safe_pos(x):
        return torch.clamp(x.float(), min=1e-5)

    def safe_log(x):
        return torch.log(safe_pos(x))

    def safe_ohr(rho):
        return torch.pow(torch.clamp(1 - rho ** 2, min=1e-5), -0.5)

    # ---------- 多模态（带 maneuver） ----------
    if use_maneuvers:
        assert isinstance(fut_pred, list), "[ERROR] fut_pred must be list of tensors when maneuvers are used"

        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1],
                          num_lon_classes * num_lat_classes, device=fut.device)
        count = 0
        for k in range(num_lon_classes):
            for l in range(num_lat_classes):
                wts = lat_pred[:, l] * lon_pred[:, k]                 # [B]
                wts = wts.unsqueeze(0).repeat(fut_pred[0].shape[0], 1)  # [T, B]

                y_pred = fut_pred[k * num_lat_classes + l]  # [T, B, 5]
                y_gt   = fut

                muX = y_pred[:, :, 0]
                muY = y_pred[:, :, 1]
                sigX = safe_pos(y_pred[:, :, 2])
                sigY = safe_pos(y_pred[:, :, 3])
                rho  = torch.tanh(y_pred[:, :, 4].float())
                ohr  = safe_ohr(rho)

                x = y_gt[:, :, 0]
                y = y_gt[:, :, 1]

                nll = -(0.5 * ohr ** 2 * (
                          sigX ** 2 * (x - muX) ** 2 +
                          sigY ** 2 * (y - muY) ** 2 -
                          2 * rho * sigX * sigY * (x - muX) * (y - muY)
                        ) - safe_log(sigX * sigY * ohr)
                        + math.log(2 * math.pi))

                acc[:, :, count] = nll + safe_log(wts)
                count += 1

        acc = -logsumexp(acc, dim=2)
        acc = acc * op_mask[:, :, 0]

        if avg_along_time:
            return torch.sum(acc) / torch.sum(op_mask[:, :, 0])
        elif separately:
            return acc, op_mask[:, :, 0]
        else:
            return torch.sum(acc, dim=1), torch.sum(op_mask[:, :, 0], dim=1)

    # ---------- 单模态 ----------
    else:
        assert isinstance(fut_pred, torch.Tensor), "[ERROR] fut_pred must be tensor when use_maneuvers=False"

        acc   = torch.zeros_like(op_mask[:, :, 0:1])
        y_pred = fut_pred
        y_gt   = fut

        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        sigX = safe_pos(y_pred[:, :, 2])
        sigY = safe_pos(y_pred[:, :, 3])
        rho  = torch.tanh(y_pred[:, :, 4].float())
        ohr  = safe_ohr(rho)

        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]

        nll = (0.5 * ohr ** 2 * (
                 sigX ** 2 * (x - muX) ** 2 +
                 sigY ** 2 * (y - muY) ** 2 -
                 2 * rho * sigX * sigY * (x - muX) * (y - muY)
               ) - safe_log(sigX * sigY * ohr)
               + math.log(2 * math.pi))

        acc[:, :, 0] = nll
        acc = acc * op_mask[:, :, 0:1]

        if avg_along_time:
            return torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
        elif separately:
            return acc[:, :, 0], op_mask[:, :, 0]
        else:
            return torch.sum(acc[:, :, 0], dim=1), torch.sum(op_mask[:, :, 0], dim=1)

def idm_loss_fn(pred_pos, true_pos, mask):
    """
    pred_pos: [B, T, 2]
    true_pos: [B, T, 2]
    mask:     [B, T] or [B, T, 1]
    """
    if mask.dim() == 3:
        mask = mask[..., 0]  # squeeze 最后一维

    diff = (pred_pos - true_pos) ** 2       # [B, T, 2]
    error = torch.sum(diff, dim=-1)         # [B, T]
    error = error * mask                    # apply mask
    return torch.sum(error) / torch.sum(mask)

def idm_accel_torch(params, s, v, dv, v_pre):
    """
    输入 shape = [B, T]，返回 [B, T]
    params = (v0,T,s0,a_max,b,delta)  –  6 个 float / tensor
    """
    v0, T, s0, a_max, b, delta = params
    s  = torch.clamp(s, min=0.1)
    s_star = s0 + v*T + v*dv/(2*torch.sqrt(a_max*b))
    acc = a_max * (1 - (v/v0).pow(delta) - (s_star/s).pow(2))
    return torch.clamp(acc, -5.0, 3.0)

def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc) / torch.sum(mask)
    return lossVal


def MAPE(y_true, y_pred, null_val=0):
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)

        mask = mask.astype("float32")
        mask /= np.mean(mask)
        mape = np.abs(np.divide((y_pred - y_true).astype("float32"), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def maskedMSETest(y_pred, y_gt, mask, separately=False):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    if separately:
        return acc[:, :, 0], mask[:, :, 0]
    else:
        lossVal = torch.sum(acc[:, :, 0], dim=1)
        counts = torch.sum(mask[:, :, 0], dim=1)
        return lossVal, counts


def maskedMAPETest(y_pred, y_gt, mask, separately=False):
    null_val = 0
    acc = torch.zeros_like(mask)

    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]

    out = torch.abs(torch.divide((x - muX), x)) + torch.abs(torch.divide((y - muY), y))
    out_0 = torch.zeros_like(out)
    out = torch.where(out > 0.75, out_0, out)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask

    if separately:
        return acc[:, :, 0], mask[:, :, 0]
    else:
        lossVal = torch.sum(acc[:, :, 0], dim=1)
        counts = torch.sum(mask[:, :, 0], dim=1)
        return lossVal, counts


def TTC_test(plan_veh_real, tar_veh_pred, tar_veh_real, tar_count):
    ttc_threshold = 3
    batch_size = 64
    delta_t = 0.2
    target_count = 0
    sum_count = 0
    ttc_count = 0
    for num in range(len(tar_count)):
        plan_veh_x = plan_veh_real[1, num, 0]
        plan_veh_y = plan_veh_real[1, num, 1]
        plan_speed = (plan_veh_real[1, num, 1] - plan_veh_real[0, num, 1])/delta_t
        for target1 in range(tar_count[num]-1):
            target1_x = tar_veh_pred[1, target_count+target1, 0] + tar_veh_real[0, target_count+target1, 0]
            target1_y = tar_veh_pred[1, target_count+target1, 1] + tar_veh_real[0, target_count+target1, 1]
            target1_speed = (tar_veh_pred[1, target_count+target1, 1] - tar_veh_pred[0, target_count+target1, 1]) / delta_t
            if TTC_judge(plan_veh_x, plan_veh_y, plan_speed, target1_x, target1_y, target1_speed, ttc_threshold):
                ttc_count += 1
                sum_count += 1
            else:
                sum_count += 1
            for target2 in range(target1+1, tar_count[num]):
                target2_x = tar_veh_pred[1, target_count + target2, 0] + tar_veh_real[0, target_count + target2, 0]
                target2_y = tar_veh_pred[1, target_count + target2, 1] + tar_veh_real[0, target_count + target2, 1]
                target2_speed = (tar_veh_pred[1, target_count + target2, 1] - tar_veh_pred[0, target_count + target2, 1]) / delta_t
                if TTC_judge(target1_x, target1_y, target1_speed, target2_x, target2_y, target2_speed, ttc_threshold):
                    ttc_count += 1
                    sum_count += 1
                else:
                    sum_count += 1
        target_count += tar_count[num]
    ttc_rate = ttc_count / sum_count

    return ttc_rate

# pylint: disable=too-many-arguments,huawei-too-many-arguments
def TTC_judge(veh1_x, veh1_y, veh1_speed, veh2_x, veh2_y, veh2_speed, ttc_threshold):
    if abs(veh1_x-veh2_x) > 10:
        return False
    if veh1_y > veh2_y:
        if veh1_speed >= veh2_speed:
            return False
        else:
            ttc = (veh1_y - veh2_y)/(veh2_speed - veh1_speed)
            if ttc <= ttc_threshold:
                return True
            else:
                return False
    else:
        if veh2_speed >= veh1_speed:
            return False
        else:
            ttc = (veh2_y - veh1_y)/(veh1_speed - veh2_speed)
            if ttc <= ttc_threshold:
                return True
            else:
                return False






## Helper function for log sum exp calculation:
def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    # Get the maximal probability value from 6 full path
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    # here (inputs - s) is to compare the relative probability with the most probable behavior.

    # and then sum up all candidate behaviors.
    # s->logP(Y | m_max,X), inputs->logP(m_i,Y | X), (inputs - s)->logP(m_i | X)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs