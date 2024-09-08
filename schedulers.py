import torch
import math
from functools import partial

def _get_schedule_with_warmup_constant_decay_lr_lambda(current_step: int, *, 
                                                 num_warmup_steps: int, 
                                                 num_stable_steps: int, 
                                                 num_decay_steps: int,
                                                 lr_start: float = 0,
                                                 lr_end: float = 0
                                                 ):
    if current_step < num_warmup_steps:
        return lr_start + float(current_step) * (1 - lr_start) / float(max(1.0, num_warmup_steps))
    elif current_step < num_warmup_steps + num_stable_steps:
        return 1.0
    else:
        decay_steps = current_step - (num_warmup_steps + num_stable_steps)
        return 1 + float(decay_steps) * ((lr_end - 1) / max(1.0, num_decay_steps))


def get_schedule_with_warmup_constant_decay(optimizer, 
                                            total_steps: int,               #总step数
                                            warmup_pct: float = 0.2,        #warmup占总step的百分比
                                            decay_pct: float = 0.2,         #decay占总step的百分比
                                            lr_start_div_factor: int = 20,  #初始lr，等于 lr_max * 1 / lr_start_div_factor
                                            lr_end_div_factor: int = 50,    #结束lr，等于 lr_max * 1 / lr_end_div_factor
                                            last_epoch: int = -1
):
    num_warmup_steps = total_steps * warmup_pct
    num_decay_steps = total_steps * decay_pct
    num_stable_steps = total_steps - num_warmup_steps - num_decay_steps
    lr_start = 1 / lr_start_div_factor
    lr_end = 1 / lr_end_div_factor
    lr_lambda = partial(_get_schedule_with_warmup_constant_decay_lr_lambda, 
                        num_warmup_steps=num_warmup_steps, 
                        num_stable_steps=num_stable_steps, 
                        num_decay_steps=num_decay_steps,
                        lr_start=lr_start,
                        lr_end=lr_end
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
# # demo:创建LambdaLR调度器
# lr_scheduler = get_schedule_with_warmup_constant_decay(optimizer, 
#                                                        total_steps=2900, 
#                                                        warmup_pct=0.15, 
#                                                        lr_end_div_factor=100,
#                                                        last_epoch=-1)


# https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
def get_cosine_anneal(eta_min):
    eta_max = 1.0
    T_max = 4
    T_cur = 1
    eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(T_cur / T_max * math.pi))
    return eta_t


def _get_schedule_with_WCD_cosine_anneal_lr_lambda(current_step: int, *, 
                                                 num_warmup_steps: int, 
                                                 num_stable_steps: int, 
                                                 num_decay_steps: int,
                                                 eta_min: float,
                                                 lr_start: float = 0,
                                                 lr_end: float = 0
                                                 ):
    if current_step < num_warmup_steps:
        return lr_start + float(current_step) * (1 - lr_start) / float(max(1.0, num_warmup_steps))
    elif current_step < num_warmup_steps + num_stable_steps/2:
        return 1.0
    elif current_step < num_warmup_steps + num_stable_steps:
        return get_cosine_anneal(eta_min)
    else:
        last_lr = get_cosine_anneal(eta_min)
        decay_steps = current_step - (num_warmup_steps + num_stable_steps)
        return last_lr + float(decay_steps) * ((lr_end - last_lr) / max(1.0, num_decay_steps))


def get_schedule_with_WCD_cosine_anneal(optimizer, 
                                        total_steps: int,               #总step数
                                        eta_min: float = 1e-6,          #最小lr
                                        warmup_pct: float = 0.2,        #warmup占总step的百分比
                                        decay_pct: float = 0.2,         #decay占总step的百分比
                                        lr_start_div_factor: int = 20,  #初始lr，等于 lr_max * 1 / lr_start_div_factor
                                        lr_end_div_factor: int = 50,    #结束lr，等于 lr_max * 1 / lr_end_div_factor
                                        last_epoch: int = -1
):
    num_warmup_steps = total_steps * warmup_pct
    num_decay_steps = total_steps * decay_pct
    num_stable_steps = total_steps - num_warmup_steps - num_decay_steps
    lr_start = 1 / lr_start_div_factor
    lr_end = 1 / lr_end_div_factor
    lr_lambda = partial(_get_schedule_with_WCD_cosine_anneal_lr_lambda, 
                        num_warmup_steps=num_warmup_steps, 
                        num_stable_steps=num_stable_steps, 
                        num_decay_steps=num_decay_steps,
                        eta_min=eta_min,
                        lr_start=lr_start,
                        lr_end=lr_end
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


# optimizer = torch.optim.AdamW(model.parameters(), lr=9e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
# # 创建LambdaLR调度器
# lr_scheduler = get_schedule_with_WCD_cosine_anneal(optimizer, 
#                                                     total_steps=2900, 
#                                                     warmup_pct=0.15, 
#                                                     lr_end_div_factor=100,
#                                                     last_epoch=-1)