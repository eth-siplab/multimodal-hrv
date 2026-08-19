# encoding=utf-8
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from trainer import *
import torch
import torch.nn as nn
# ---- DDP-aware DataLoader helper ----
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader  
#
from utils import *
import argparse
from datetime import datetime
import pickle
import os, numpy as np, torch, torch.nn as nn, torch.distributed as dist
import logging
import sys
from copy import deepcopy
import fitlog
import random

parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--cuda', default=0, type=int, help='cuda device ID, 0/1')
# hyperparameter
parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
parser.add_argument('--n_epoch', type=int, default=60, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
# dataset
parser.add_argument('--dataset', type=str, default='wildppg', choices=['wildppg', 'ieee_small','bidmc', 'dalia'], help='name of dataset')
parser.add_argument('--n_feature', type=int, default=77, help='name of feature dimension')
parser.add_argument('--len_sw', type=int, default=30, help='length of sliding window')
parser.add_argument('--n_class', type=int, default=18, help='number of class')
parser.add_argument('--cases', type=str, default='subject_val', choices=['random', 'subject', 'subject_large', 'cross_device', 'joint_device'], help='name of scenarios')
parser.add_argument('--split_ratio', type=float, default=0.2, help='split ratio of test/val: train(0.64), val(0.16), test(0.2)')
parser.add_argument('--target_domain', type=str, default='0')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
# models
parser.add_argument('--backbone', type=str, default='DCL', choices=['FCN', 'FCN_b', 'DCL', 'LSTM', 'Transformer', 'resnet', 'TWaveNet','unet'], help='name of framework')
parser.add_argument('--out_dim', type=int, default=128, help='output dimension of the encoder')
# ssm
parser.add_argument('--ssm_hidden', type=int, default=64, help='hidden dimension of SSM head')
parser.add_argument('--ss_feat_dim', type=int, default=3, help='feature dimension of SSM head')
parser.add_argument('--seq_batch_size', type=int, default=256, help='learning rate of SSM head')
# 
parser.add_argument('--quality_mask', action='store_true', help='whether to use quality mask')
parser.add_argument('--keep_ratio', type=float, default=0.80, help='keep ratio if quality_tau is None')
# model parameters
parser.add_argument('--block', type=int, default=8, help='number of groups')
parser.add_argument('--stride', type=int, default=2, help='stride')
# log
parser.add_argument('--logdir', type=str, default='log/', help='log directory')
parser.add_argument('--no_ssm', action='store_true', help='skip Stage-2 SSM and evaluate UNet only')
parser.add_argument('--stage2_head', type=str, default='ssm', choices=['ssm', 'ctf'],
                    help='Stage-2 head: SSM (kalman AR2) or causal transformer ablation')

# python main.py --dataset 'wildppg' --backbone 'unet' --block 8 --lr 5e-4 --n_epoch 999 --cuda 0 

############### Parser done ################
def is_dist():
    return dist.is_available() and dist.is_initialized()

def dist_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()    

def is_rank0():
    return (not is_dist()) or dist.get_rank() == 0

def _norm_sd(model):
    return (model.module.state_dict() if isinstance(model, DDP) else model.state_dict())    

def device_from_env(args):
    # prefer LOCAL_RANK if using torchrun
    lr = int(os.environ.get("LOCAL_RANK", args.cuda if hasattr(args, "cuda") else 0))
    torch.cuda.set_device(lr)
    return torch.device(f"cuda:{lr}" if torch.cuda.is_available() else "cpu")

def rewrap_with_sampler(loader, shuffle=False):
    if not is_dist():
        return loader, None
    ds = loader.dataset
    sampler = DistributedSampler(ds, shuffle=shuffle, drop_last=True)  # <-- True
    new_loader = DataLoader(
        ds,
        batch_size=loader.batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=True,                       
        num_workers=loader.num_workers,
        pin_memory=getattr(loader, "pin_memory", True),
        persistent_workers=getattr(loader, "persistent_workers", False),
        prefetch_factor=getattr(loader, "prefetch_factor", None),
        collate_fn=loader.collate_fn,
    )
    return new_loader, sampler

# def heteroscedastic_nll(mu, logvar, target, var_reg=1e-4):
#     nll = 0.5 * (torch.exp(-logvar) * (mu - target)**2 + logvar)
#     return nll.mean() + var_reg * (logvar**2).mean()

#
def mae_or_nll(args, mu, logvar, target):
    mu = mu.reshape(-1)
    target = target.reshape(-1)

    valid = torch.isfinite(mu) & torch.isfinite(target)
    if valid.sum() == 0:
        # 0 loss that keeps graph valid
        return (mu.sum() * 0.0)

    mu_v = mu[valid].float()
    tgt_v = target[valid].float()

    if args.backbone.lower() == "dcl":
        # IMPORTANT: SmoothL1 must not see NaNs
        return torch.nn.functional.smooth_l1_loss(mu_v, tgt_v)

    # keep existing NLL path (already masks inside too, but ok)
    return heteroscedastic_nll(mu_v, logvar.reshape(-1)[valid].float(), tgt_v)

#
def heteroscedastic_nll(mu, logvar, target, var_reg=1e-4):
    # flatten for safe masking
    mu = mu.reshape(-1)
    logvar = logvar.reshape(-1)
    target = target.reshape(-1)

    valid = torch.isfinite(mu) & torch.isfinite(logvar) & torch.isfinite(target)
    if valid.sum() == 0:
        # return a 0 loss that still participates in autograd
        return (mu.sum() * 0.0)

    mu = mu[valid]
    logvar = logvar[valid]
    target = target[valid]

    # clamp to prevent exp overflow/underflow
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)

    nll = 0.5 * (torch.exp(-logvar) * (mu - target) ** 2 + logvar)
    return nll.mean() + var_reg * (logvar ** 2).mean()

# ---------- helpers ----------

# ---------- training ----------

def make_loader(dataset, args, shuffle, ddp=True, collate_fn=None):
    sampler = None
    if ddp and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=False,
        sampler=sampler,
        num_workers=getattr(args, "num_workers", 8),
        pin_memory=True,
        persistent_workers=True if getattr(args, "num_workers", 0) > 0 else False,
        prefetch_factor=4 if getattr(args, "num_workers", 0) > 0 else None,
        collate_fn=collate_fn,
    ), sampler

def train(args, train_loader, val_loader, train_sampler, model, DEVICE, criterion, save_dir='results/'):
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    use_fused = "fused" in nn.__dict__.get("AdamW", torch.optim.AdamW).__init__.__code__.co_varnames
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                  fused=use_fused) if use_fused else torch.optim.AdamW(
                                  model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=15, factor=0.5, min_lr=1e-7, verbose=False
    )

    best_sd = _norm_sd(model)                     # normalized (no 'module.' prefix)
    min_val_loss, counter = float('inf'), 0
    ckpt_path = os.path.join(save_dir, args.model_name + '.pt')

    for epoch in range(args.n_epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        for sample, target in train_loader:
            (ppg, imu, temp) = sample
            ppg  = ppg.to(DEVICE, non_blocking=True).float()
            imu  = imu.to(DEVICE, non_blocking=True).float()
            temp = temp.to(DEVICE, non_blocking=True).float()
            target = target.to(DEVICE, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            if args.dataset == 'bidmc':
                with  torch.cuda.amp.autocast(enabled=False):
                    mu, logvar = model(ppg)
            else:
                with  torch.cuda.amp.autocast(enabled=False):
                    mu, logvar = model(ppg, imu, temp)

            # compute loss in fp32
            loss = criterion(mu.float(), logvar.float(), target.float())         

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        # ---- validation (rank-0 computes) ----
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_mae  = 0.0
            n_batches = 0
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                for sample, target in val_loader:
                    (ppg, imu, temp) = sample
                    ppg  = ppg.to(DEVICE, non_blocking=True).float()
                    imu  = imu.to(DEVICE, non_blocking=True).float()
                    temp = temp.to(DEVICE, non_blocking=True).float()
                    target = target.to(DEVICE, non_blocking=True).float()

                    mu, logvar = model(ppg, imu, temp) if args.dataset != 'bidmc' else model(ppg)  # bidmc has no imu/temp
                    batch_loss = criterion(mu.float(), logvar.float(), target)
                    val_loss += batch_loss.item()
                    val_mae  += torch.mean(torch.abs(mu - target)).item()
                    n_batches += 1

            if n_batches > 0:
                val_loss /= n_batches
                val_mae  /= n_batches

            # reduce val_loss across ranks
            t = torch.tensor([val_loss], device=DEVICE)
            if dist.is_initialized():
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                val_loss = t.item()

            if is_rank0():
                if np.isfinite(val_loss) and val_loss <= min_val_loss:
                    min_val_loss = val_loss
                    best_sd = _norm_sd(model)
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(best_sd, ckpt_path)
                    counter = 0
                else:
                    counter += 1
                if np.isfinite(val_loss):
                    scheduler.step(val_loss)

            if dist.is_initialized():
                dist.barrier()
                stop = torch.tensor([1 if counter > 90 else 0], device=DEVICE)
                dist.broadcast(stop, src=0)
                if stop.item() == 1:
                    break
        dist_barrier()

    # broadcast best_sd so ALL ranks return a valid dict (no disk read needed)
    if dist.is_initialized():
        obj_list = [best_sd if is_rank0() else None]
        dist.broadcast_object_list(obj_list, src=0)
        best_sd = obj_list[0]

    return best_sd

def test(test_loader, model, DEVICE, _criterion_unused=None, plot=False):
    model.eval()
    preds, targs = [], []

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        for sample, target in test_loader:
            (ppg, imu, temp) = sample
            ppg  = ppg.to(DEVICE, non_blocking=True).float()
            imu  = imu.to(DEVICE, non_blocking=True).float()
            temp = temp.to(DEVICE, non_blocking=True).float()
            target = target.to(DEVICE, non_blocking=True).float()

            mu, logvar = model(ppg, imu, temp) if args.dataset != 'bidmc' else model(ppg)  # bidmc has no imu/temp
            preds.append(mu.detach().cpu())
            targs.append(target.detach().cpu())

    pred = torch.cat(preds, 0)
    targ = torch.cat(targs, 0)

    mse  = torch.mean((pred - targ) ** 2).item()
    rmse = float(np.sqrt(mse))
    mae  = torch.mean(torch.abs(pred - targ)).item()
    p_np, t_np = pred.numpy(), targ.numpy()
    r = 0.0 if (np.std(p_np) < 1e-8 or np.std(t_np) < 1e-8) else float(np.corrcoef(t_np, p_np)[0, 1])
    return rmse, mae, r

# ------------------------------
# Orchestration
def train_sup(args):
    train_loader, val_loader, test_loader = setup_dataloaders(args)
    DEVICE = device_from_env(args)
    train_loader, train_sampler = rewrap_with_sampler(train_loader, shuffle=False)

    model = build_model(args).to(DEVICE)

    # Only use DDP when actually multi-process; avoids reducer issues in 1-GPU runs
    if is_dist() and dist.get_world_size() > 1:
        model = DDP(
            model,
            device_ids=[DEVICE.index],
            gradient_as_bucket_view=True,
            find_unused_parameters=True,  # tolerate data-dependent branches / optional modalities
        )

    args.model_name = f"{args.backbone}_{args.dataset}_cuda{args.cuda}_bs{args.batch_size}_sw{args.len_sw}"
    save_dir = 'results/'
    if is_rank0():
        os.makedirs(save_dir, exist_ok=True); os.makedirs(args.logdir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, args.model_name + '.pt')

    criterion = lambda mu, logvar, target: mae_or_nll(args, mu, logvar, target)
    try:
        best_sd = train(args, train_loader, val_loader, train_sampler, model, DEVICE, criterion, save_dir=save_dir)
    except TypeError:
        best_sd = train(args, train_loader, val_loader, train_sampler, model, DEVICE, criterion, save_dir=save_dir)

    if isinstance(model, DDP):
        model.module.load_state_dict(best_sd)
    else:
        model.load_state_dict(best_sd)

    if is_rank0():
        torch.save(best_sd, ckpt_path)

    if is_dist(): dist.barrier()

    # Stage-1 Eval
    if is_rank0():
        rmse, mae, corr = test(test_loader, model, DEVICE, criterion, plot=False)
    else:
        rmse = mae = corr = float('nan')
    if is_dist(): dist.barrier()

    # ---- subjectwise eval + early return ----
    if getattr(args, "no_ssm", False):
        setattr(args, 'cases', 'ssm_train')  # reuse ordered_test builder
        _seq_train_loader_unused, ordered_test = setup_ssm_dataloaders(args)

        unet_frozen = model.module if isinstance(model, DDP) else model
        for p in unet_frozen.parameters():
            p.requires_grad = False
        unet_frozen.eval()

        results = eval_subjectwise_unet_only(
            unet_frozen, ordered_test, DEVICE,
            unet_amp=False,
            quality_mask=False,
            keep_ratio=getattr(args, "keep_ratio", 0.80),
            quality_weights=getattr(args, "quality_weights", (1.0, 0.5, 0.5, 0.0)),
            quality_tau=getattr(args, "quality_tau", None),
        )
        return results

    # ----- Stage 2: freeze the model, train SSM head on short sequential chunks -----
    # Seq loaders (return windows [B,T,C,W], feats [B,T,F], rr_ecg [B,T], mask [B,T])
    setattr(args, 'cases', 'ssm_train')   # new case for ssm training
    seq_train_loader, ordered_test = setup_ssm_dataloaders(args)

    # Unwrap if DDP, freeze the model
    unet_frozen = model.module if isinstance(model, DDP) else model
    for p in unet_frozen.parameters(): p.requires_grad = False
    unet_frozen.eval()

    ssm_head = setup_SSM(args, DEVICE)

    opt = torch.optim.Adam(ssm_head.parameters(), lr=getattr(args, "ssm_lr", 1e-3), weight_decay=1e-6)
    best_val = float('inf')
    best_ssm = None

    # print once before the loop
    if is_rank0():
        with torch.no_grad():
            windows, feats, rr_ecg, mask = next(iter(seq_train_loader))
            windows = windows.to(DEVICE); feats = feats.to(DEVICE)
            rr_ecg  = rr_ecg.to(DEVICE);  mask  = mask.to(DEVICE)
            m_seq, logvar = forward_unet_seq(unet_frozen, windows)
            print(
                "[Stage2 Sanity]",
                "med rr_ecg(ms)=", float(torch.nanmedian(rr_ecg)),
                "med m_seq(ms)=",  float(torch.nanmedian(m_seq)),
                "R_med=",          float(torch.nanmedian(torch.exp(logvar)))
            )    

    warmup_epochs = 3
    E = getattr(args, "stage2_epochs", 15)
    # E = getattr(args, "stage2_epochs", 2) # quick test
    for epoch in range(E):
        warmup_w = 1.0 if epoch < warmup_epochs else 0.0   # 1.0 = strong warm-up, 0.0 = fully learned
        train_loss = train_stage2_epoch(
            seq_train_loader, unet_frozen, ssm_head, opt, DEVICE,
            epoch=epoch, warmup_epochs=warmup_epochs  # optional, for logging
        )
        if is_rank0():
            print(f"[Stage2][{epoch+1}/{E}] train={train_loss:.3f}")

    # (optional) save the **last** SSM checkpoint for reproducibility
    if is_rank0():
        last_sd = (ssm_head.module if hasattr(ssm_head, "module") else ssm_head).state_dict()
        torch.save(last_sd, os.path.join(save_dir, args.model_name + "_ssm_last.pt"))

    if dist.is_initialized():
        dist.barrier()  # keep ranks in sync

    # Final evaluation with SSM head on a sequential test loader
    results = eval_subjectwise(unet_frozen, ssm_head, ordered_test, DEVICE,
                              quality_mask=getattr(args, "quality_mask", False), 
                              keep_ratio=getattr(args, "keep_ratio", 0.80),
                              quality_weights=getattr(args, "quality_weights", (1.0,0.5,0.5,0.0)),
                              quality_tau=getattr(args, "quality_tau", None),
                              )   
    return results

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_domains(args):
    # don’t parse again here; args already parsed in __main__
    if args.dataset == 'wildppg' or args.dataset == 'dalia':
        return [0, 1, 2, 3, 4]  # 5 fold CV
    elif args.dataset == 'bidmc':
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]           # 10 fold CV
    return [0]

def dist_init():
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, dist.get_world_size() if dist.is_initialized() else 1, device

if __name__ == '__main__':
    args = parser.parse_args()

    # DDP init
    local_rank, world_size, DEVICE = dist_init()
    args.cuda = local_rank  
    if is_rank0():
        print(f'device: {DEVICE}  dataset: {args.dataset}  world_size: {world_size}')

    # use more loader workers per GPU if you want
    if not hasattr(args, 'num_workers'):
        args.num_workers = 8

    domain = set_domains(args)  # e.g., [0,1,2,3,4]
    all_metrics = []

    for i in range(1):  # one seed
        set_seed(i * 10 + 1 + local_rank)
        if is_rank0():
            print(f'Training for seed {i}')

        seed_metric = []
        for k in domain:
            setattr(args, 'target_domain', str(k))     # or list of subject ids
            setattr(args, 'save', args.dataset + str(k))
            setattr(args, 'cases', 'subject_val')

            # --- train_sup returns the subject-wise results dict ONLY ---
            subj_results = train_sup(args)

            if is_rank0():
                model_tag = f"{args.backbone}_{args.dataset}_cuda{args.cuda}_bs{args.batch_size}_sw{args.len_sw}"
                fold_dir  = os.path.join("results", model_tag, f"fold_{k}")

                # NEW: ratio subfolder
                ratio = float(getattr(args, "keep_ratio", 0.80))
                ratio_tag = f"ratio_{ratio:.2f}"
                out_dir = os.path.join(fold_dir, ratio_tag)
                os.makedirs(out_dir, exist_ok=True)

                # Save per-subject arrays + CSV summary (add ratio to CSV tag too)
                csv_path = save_subject_results(subj_results, out_dir, fold_tag=f"fold{k}_r{ratio:.2f}")
                print(f"[Fold {k}] Subject-wise results saved to: {out_dir}")
                print(f"[Fold {k}] Summary CSV: {csv_path}")

                # Aggregate metrics across subjects for reporting
                agg = aggregate_results(subj_results)
                print(f"[Fold {k}] agg RMSE={agg['RMSE']:.4f}  MAE={agg['MAE']:.4f}  r={agg['r']:.4f}  Bias={agg['Bias']:.4f}")
                seed_metric.append([agg["RMSE"], agg["MAE"], agg["r"]])

            # keep ranks in sync each fold
            if dist.is_initialized():
                dist.barrier()

        if is_rank0():
            seed_metric = np.array(seed_metric, dtype=float)  # shape [num_folds, 3]
            all_metrics.append([
                np.mean(seed_metric[:, 0]),  # RMSE
                np.mean(seed_metric[:, 1]),  # MAE
                np.mean(seed_metric[:, 2]),  # r
            ])

    if is_rank0():
        values = np.array(all_metrics, dtype=float)
        mean = np.mean(values, 0)
        std  = np.std(values, 0)
        print('RMSE: {:.3f}, MAE: {:.4f}, r: {:.4f}'.format(mean[0], mean[1], mean[2]))
        print('Std RMSE: {:.3f}, Std MAE: {:.4f}, Std r: {:.4f}'.format(std[0], std[1], std[2]))

    # tidy teardown
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()