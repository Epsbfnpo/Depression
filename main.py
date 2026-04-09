import argparse
import random
import os
import copy
import numpy as np
import yaml
import wandb
import torch
from tqdm import tqdm
from models import DepMamba
from datasets import get_dvlog_dataloader, get_dvlog_kfold_loaders, get_lmvd_dataloader

CONFIG_PATH = "./config/config.yaml"



def seed_everything(seed=42):
    """Set random seeds for full experiment reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 基础的 cudnn 确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 【新增防线】强制 PyTorch 使用确定性算法，并固定 CUDA 工作区配置
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

def parse_args():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    parser = argparse.ArgumentParser(description="Train and test a model.")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--train_gender", type=str)
    parser.add_argument("--test_gender", type=str)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-bs", "--batch_size", type=int)
    parser.add_argument("-lr", "--learning_rate", type=float)
    parser.add_argument("-ds", "--dataset", type=str)
    parser.add_argument("-g", "--gpu", type=str, default="0", help="GPU id(s), e.g. '0' or '0,1' or 'cuda:0,1' or 'cpu'")
    parser.add_argument("-wdb", "--if_wandb", type=bool, default=False)
    parser.add_argument("-tqdm", "--tqdm_able", type=bool)
    parser.add_argument("-tr", "--train", action="store_true")
    parser.set_defaults(**config)
    args = parser.parse_args()
    return args

def _parse_gpu_arg(gpu_arg: str):
    s = str(gpu_arg).strip().lower()
    if s in ("cpu", "none", "-1", ""):
        return None
    s = s.replace("cuda:", "")
    ids = [int(x) for x in s.split(",") if x != ""]
    return ids

def _compute_binary_metrics(pred, y):
    tp = torch.sum((pred == 1) & (y == 1)).item()
    fp = torch.sum((pred == 1) & (y == 0)).item()
    tn = torch.sum((pred == 0) & (y == 0)).item()
    fn = torch.sum((pred == 0) & (y == 1)).item()
    return tp, fp, tn, fn

def _finalize_metrics(running_loss, sample_count, tp, fp, tn, fn):
    l = running_loss / sample_count
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0)
    accuracy = ((tp + tn) / sample_count if sample_count > 0 else 0.0)
    pred_pos = tp + fp
    pred_neg = tn + fn
    return {
        "loss": l,
        "acc": accuracy,
        "f1": f1_score,
        "pred_dist": f"Pos:{pred_pos}|Neg:{pred_neg}",
        "recall": recall,
        "precision": precision,
    }

def _get_model_for_checks(net):
    return net.module if isinstance(net, torch.nn.DataParallel) else net

def _run_silent_failure_check(net):
    model_for_check = _get_model_for_checks(net)
    if hasattr(model_for_check, "reset_silent_failure_flags"):
        model_for_check.reset_silent_failure_flags()
    return model_for_check

def _assert_silent_failure_check(model_for_check):
    if hasattr(model_for_check, "assert_new_path_executed"):
        model_for_check.assert_new_path_executed()

def train_epoch(net, train_loader, loss_fn, optimizer, device, current_epoch, total_epochs, tqdm_able, accumulation_steps=16):
    net.train()
    accumulation_steps = max(1, int(accumulation_steps))
    sample_count = 0
    running_loss = 0.
    TP, FP, TN, FN = 0, 0, 0, 0
    optimizer.zero_grad()
    with tqdm(train_loader, desc=f"Training epoch {current_epoch}/{total_epochs}", leave=False, unit="batch", disable=tqdm_able) as pbar:
        for i, batch in enumerate(pbar):
            model_for_check = _run_silent_failure_check(net)
            videos = batch["video"].to(device)
            audios = batch["audio"].to(device)
            y = batch["label"].to(device)
            y_pred = net(audios, videos)
            _assert_silent_failure_check(model_for_check)
            loss = loss_fn(y_pred, y.long())
            (loss / accumulation_steps).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            sample_count += videos.shape[0]
            running_loss += loss.item() * videos.shape[0]
            pred = torch.argmax(y_pred, dim=1)
            tp, fp, tn, fn = _compute_binary_metrics(pred, y)
            TP += tp
            FP += fp
            TN += tn
            FN += fn
            current = _finalize_metrics(running_loss, sample_count, TP, FP, TN, FN)
            pbar.set_postfix(current)
    return _finalize_metrics(running_loss, sample_count, TP, FP, TN, FN)

def val(net, val_loader, loss_fn, device, tqdm_able):
    net.eval()
    sample_count = 0
    running_loss = 0.
    TP, FP, TN, FN = 0, 0, 0, 0
    with torch.no_grad():
        with tqdm(val_loader, desc="Validating", leave=False, unit="batch", disable=tqdm_able) as pbar:
            for batch in pbar:
                model_for_check = _run_silent_failure_check(net)
                videos = batch["video"].to(device)
                audios = batch["audio"].to(device)
                y = batch["label"].to(device)
                y_pred = net(audios, videos)
                _assert_silent_failure_check(model_for_check)
                loss = loss_fn(y_pred, y.long())
                sample_count += videos.shape[0]
                running_loss += loss.item() * videos.shape[0]
                pred = torch.argmax(y_pred, dim=1)
                tp, fp, tn, fn = _compute_binary_metrics(pred, y)
                TP += tp
                FP += fp
                TN += tn
                FN += fn
                pbar.set_postfix(_finalize_metrics(running_loss, sample_count, TP, FP, TN, FN))
    return _finalize_metrics(running_loss, sample_count, TP, FP, TN, FN)

def main():
    args = parse_args()
    gpu_ids = _parse_gpu_arg(args.gpu)
    if gpu_ids is None or not torch.cuda.is_available():
        primary_device = torch.device("cpu")
        dp_device_ids = None
    else:
        torch.cuda.set_device(gpu_ids[0])
        primary_device = torch.device(f"cuda:{gpu_ids[0]}")
        dp_device_ids = gpu_ids if len(gpu_ids) > 1 else None
    print(f"[Device] primary={primary_device}, data_parallel_ids={dp_device_ids}")

    if dp_device_ids is not None:
        raise ValueError(
            "Fatal Error: Pure asynchronous architecture with Batch=1 CANNOT be split across "
            "multiple GPUs using DataParallel. Please set -g to a single GPU ID (e.g., -g '0')."
        )

    args.data_dir = os.path.join(args.data_dir, args.dataset)

    if args.if_wandb:
        wandb_run_name = f"{args.model}-{args.train_gender}-{args.test_gender}-baseline-3seeds"
        wandb.init(project="mamnba_ad", config=args, name=wandb_run_name)

    for i_iter in range(3):
        current_seed = 42 + i_iter
        seed_everything(seed=current_seed)
        print(f"\n=======================================================")
        print(f"[INFO] Starting Iteration {i_iter} with Random Seed: {current_seed}")
        print(f"=======================================================\n")

        print(args)
        os.makedirs(f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}", exist_ok=True)
        os.makedirs(f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}/samples", exist_ok=True)
        os.makedirs(f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}/checkpoints", exist_ok=True)
        
        if args.model == "DepMamba":
            if args.dataset=='lmvd':
                net = DepMamba(**args.mmmamba_lmvd)
            elif args.dataset=='dvlog':
                net = DepMamba(**args.mmmamba)
        else:
            raise NotImplementedError(f"The {args.model} method has not been implemented by this repo")
        net = net.to(primary_device)
        # 负类(0)较少，赋予较高权重；正类(1)较多，权重较小。
        # 权重计算逻辑：(Total / Negative) 和 (Total / Positive) 的归一化变形
        class_weights = torch.tensor([1.38, 1.0], dtype=torch.float32).to(primary_device)
        
        if args.dataset == 'dvlog':
            kfold_loaders = get_dvlog_kfold_loaders(
                args.data_dir,
                gender=args.train_gender,
                n_splits=5,
                random_state=current_seed,
            )
            test_loader = get_dvlog_dataloader(args.data_dir, "test", args.test_gender)
        elif args.dataset == 'lmvd':
            train_loader = get_lmvd_dataloader(args.data_dir, "train", args.train_gender)
            val_loader = get_lmvd_dataloader(args.data_dir, "valid", args.test_gender)
            test_loader = get_lmvd_dataloader(args.data_dir, "test", args.test_gender)
            
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        initial_state_dict = copy.deepcopy(net.state_dict())

        def build_optimizer_and_scheduler():
            decay_params = []
            no_decay_params = []
            for name, param in net.named_parameters():
                if not param.requires_grad:
                    continue
                if param.ndim <= 1 or "bias" in name or "norm" in name or "latent_pe" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            optim_groups = [
                {"params": decay_params, "weight_decay": 1e-1},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]
            optimizer = torch.optim.AdamW(optim_groups, lr=args.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                verbose=True
            )
            return optimizer, scheduler
        
        if args.train:
            if args.dataset == "dvlog":
                for fold_idx, (train_loader, val_loader) in enumerate(kfold_loaders):
                    print(f"[CV] Starting fold {fold_idx + 1}/5")
                    net.load_state_dict(initial_state_dict)
                    optimizer, scheduler = build_optimizer_and_scheduler()
                    best_val_loss = float('inf')
                    early_stop_patience = 10
                    early_stop_counter = 0
                    warmup_epochs = 5
                    for epoch in range(args.epochs):
                        if epoch < warmup_epochs:
                            lr_scale = min(1.0, float(epoch + 1) / warmup_epochs)
                            for pg in optimizer.param_groups:
                                pg["lr"] = lr_scale * args.learning_rate
                        train_results = train_epoch(
                            net,
                            train_loader,
                            loss_fn,
                            optimizer,
                            primary_device,
                            epoch,
                            args.epochs,
                            args.tqdm_able,
                            accumulation_steps=args.batch_size,
                        )
                        val_results = val(net, val_loader, loss_fn, primary_device, args.tqdm_able)
                        print(f"[Fold {fold_idx + 1} | Epoch {epoch + 1}] Train metrics: {train_results}")
                        print(f"[Fold {fold_idx + 1} | Epoch {epoch + 1}] Valid metrics: {val_results}")
                        current_val_loss = val_results["loss"]
                        if scheduler is not None:
                            scheduler.step(current_val_loss)
                        if current_val_loss < best_val_loss:
                            best_val_loss = current_val_loss
                            early_stop_counter = 0
                            save_path = f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}/checkpoints/best_model_fold_{fold_idx}.pt"
                            torch.save(net.state_dict(), save_path)
                            print(f"--> [Checkpoint] New best val loss: {best_val_loss:.4f} saved.")
                        else:
                            early_stop_counter += 1
                            if early_stop_counter >= early_stop_patience:
                                print(f"[EarlyStopping] Stop at epoch {epoch + 1}; best val loss={best_val_loss:.4f}")
                                break
                        if args.if_wandb:
                            wandb.log({
                                f"fold_{fold_idx + 1}/loss/train": train_results["loss"],
                                f"fold_{fold_idx + 1}/acc/train": train_results["acc"],
                                f"fold_{fold_idx + 1}/precision/train": train_results["precision"],
                                f"fold_{fold_idx + 1}/recall/train": train_results["recall"],
                                f"fold_{fold_idx + 1}/f1/train": train_results["f1"],
                                f"fold_{fold_idx + 1}/loss/val": val_results["loss"],
                                f"fold_{fold_idx + 1}/acc/val": val_results["acc"],
                                f"fold_{fold_idx + 1}/precision/val": val_results["precision"],
                                f"fold_{fold_idx + 1}/recall/val": val_results["recall"],
                                f"fold_{fold_idx + 1}/f1/val": val_results["f1"],
                                f"fold_{fold_idx + 1}/best_loss/val": best_val_loss
                            })
            else:
                optimizer, scheduler = build_optimizer_and_scheduler()
                best_val_loss = float('inf')
                early_stop_patience = 10
                early_stop_counter = 0
                warmup_epochs = 5
                for epoch in range(args.epochs):
                    if epoch < warmup_epochs:
                        lr_scale = min(1.0, float(epoch + 1) / warmup_epochs)
                        for pg in optimizer.param_groups:
                            pg["lr"] = lr_scale * args.learning_rate
                    train_results = train_epoch(
                        net,
                        train_loader,
                        loss_fn,
                        optimizer,
                        primary_device,
                        epoch,
                        args.epochs,
                        args.tqdm_able,
                        accumulation_steps=args.batch_size,
                    )
                    val_results = val(net, val_loader, loss_fn, primary_device, args.tqdm_able)
                    print(f"[Epoch {epoch + 1}] Train metrics: {train_results}")
                    print(f"[Epoch {epoch + 1}] Valid metrics: {val_results}")
                    current_val_loss = val_results["loss"]
                    if scheduler is not None:
                        scheduler.step(current_val_loss)
                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        early_stop_counter = 0
                        torch.save(net.state_dict(), f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}/checkpoints/best_model.pt")
                        print(f"--> [Checkpoint] New best val loss: {best_val_loss:.4f} saved.")
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= early_stop_patience:
                            print(f"[EarlyStopping] Stop at epoch {epoch + 1}; best val loss={best_val_loss:.4f}")
                            break
                    if args.if_wandb:
                        wandb.log({
                            "loss/train": train_results["loss"],
                            "acc/train": train_results["acc"],
                            "precision/train": train_results["precision"],
                            "recall/train": train_results["recall"],
                            "f1/train": train_results["f1"],
                            "loss/val": val_results["loss"],
                            "acc/val": val_results["acc"],
                            "precision/val": val_results["precision"],
                            "recall/val": val_results["recall"],
                            "f1/val": val_results["f1"],
                            "best_loss/val": best_val_loss
                        })
                    
        with torch.no_grad():
            if args.dataset == 'dvlog':
                print("[INFO] Evaluating K-Fold Ensemble on Test Set...")
                test_metrics_list = []
                for fold_idx in range(5):
                    model_path = f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}/checkpoints/best_model_fold_{fold_idx}.pt"
                    if os.path.exists(model_path):
                        net.load_state_dict(torch.load(model_path, map_location=primary_device))
                        net.eval()
                        fold_test_results = val(net, test_loader, loss_fn, primary_device, args.tqdm_able)
                        test_metrics_list.append(fold_test_results)
                        print(f"Fold {fold_idx + 1} Test results: {fold_test_results}")
                if len(test_metrics_list) == 0:
                    raise RuntimeError("No fold checkpoint found for DVlog evaluation.")
                test_results = {}
                for key in test_metrics_list[0].keys():
                    first_value = test_metrics_list[0][key]
                    if isinstance(first_value, (int, float, np.floating)):
                        test_results[key] = np.mean([m[key] for m in test_metrics_list])
                    else:
                        test_results[key] = "ensemble"
                print("\n[INFO] Final Ensemble Test Results (Averaged across 5 folds):")
            else:
                net.load_state_dict(torch.load(f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}/checkpoints/best_model.pt", map_location=primary_device))
                net.eval()
                test_results = val(net, test_loader, loss_fn, primary_device, args.tqdm_able)
                print("\nTest results:")
            print(test_results)
            os.makedirs("./results", exist_ok=True)
            with open(f'./results/{args.dataset}_{args.model}_{str(i_iter)}.txt','w') as f:    
                test_result_str = f'Accuracy:{test_results["acc"]}, Precision:{test_results["precision"]}, Recall:{test_results["recall"]}, F1:{test_results["f1"]}, Avg:{(test_results["acc"] + test_results["precision"]+ test_results["recall"]+ test_results["f1"])/4.0}'
                f.write(test_result_str)

            if args.if_wandb:
                if args.dataset == 'dvlog':
                    for fold_idx in range(5):
                        model_path = f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}/checkpoints/best_model_fold_{fold_idx}.pt"
                        if os.path.exists(model_path):
                            artifact = wandb.Artifact(f"best_model_iter_{i_iter}_fold_{fold_idx}", type="model")
                            artifact.add_file(model_path)
                            wandb.log_artifact(artifact)
                else:
                    artifact = wandb.Artifact(f"best_model_iter_{i_iter}", type="model")
                    artifact.add_file(f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}/checkpoints/best_model.pt")
                    wandb.log_artifact(artifact)
                wandb.log({
                    f"test_iter_{i_iter}/acc": test_results["acc"],
                    f"test_iter_{i_iter}/loss": test_results["loss"],
                    f"test_iter_{i_iter}/precision": test_results["precision"],
                    f"test_iter_{i_iter}/recall": test_results["recall"],
                    f"test_iter_{i_iter}/f1": test_results["f1"]
                })

    # 5. 【新增】在3次大循环全部结束后，安全关闭 WandB
    if args.if_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
