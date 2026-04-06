import argparse
import random
import os
import numpy as np
import yaml
import wandb
import torch
from tqdm import tqdm
from models import DepMamba
from datasets import get_dvlog_dataloader, get_lmvd_dataloader

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
    return {"loss": l, "acc": accuracy, "precision": precision, "recall": recall, "f1": f1_score}

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
    # 1. 删除了原先在这里的 seed_everything(seed=42)
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
    args.data_dir = os.path.join(args.data_dir,args.dataset)

    # 2. 【新增】将 WandB 初始化移到循环外，并加上 '-3seeds' 后缀
    if args.if_wandb:
        wandb_run_name = f"{args.model}-{args.train_gender}-{args.test_gender}-baseline-3seeds"
        wandb.init(project="mamnba_ad", config=args, name=wandb_run_name)
        args = wandb.config

    for i_iter in range(3):
        # 3. 【新增】每次循环开头动态设置严格的种子
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
        
        if dp_device_ids is not None:
            net = torch.nn.DataParallel(net, device_ids=dp_device_ids)
            
        if args.dataset=='dvlog':
            train_loader = get_dvlog_dataloader(args.data_dir, "train", 1, args.train_gender)
            val_loader = get_dvlog_dataloader(args.data_dir, "valid", 1, args.test_gender)
            test_loader = get_dvlog_dataloader(args.data_dir, "test", 1, args.test_gender)
        elif args.dataset=='lmvd':
            train_loader = get_lmvd_dataloader(args.data_dir, "train", 1, args.train_gender)
            val_loader = get_lmvd_dataloader(args.data_dir, "valid", 1, args.test_gender)
            test_loader = get_lmvd_dataloader(args.data_dir, "test", 1, args.test_gender)
            
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=5e-2)
        best_val_f1 = -1.0
        early_stop_patience = 20
        early_stop_counter = 0
        
        if args.train:
            for epoch in range(args.epochs):
                train_results = train_epoch(net, train_loader, loss_fn, optimizer, primary_device, epoch, args.epochs, args.tqdm_able)
                val_results = val(net, val_loader, loss_fn, primary_device, args.tqdm_able)
                print(f"[Epoch {epoch + 1}] Train metrics: {train_results}")
                print(f"[Epoch {epoch + 1}] Valid metrics: {val_results}")
                current_val_f1 = val_results["f1"]
                if current_val_f1 > best_val_f1:
                    best_val_f1 = current_val_f1
                    early_stop_counter = 0
                    torch.save(net.state_dict(),f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}/checkpoints/best_model.pt")
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= early_stop_patience:
                        print(f"[EarlyStopping] Stop at epoch {epoch + 1}; best val f1={best_val_f1:.4f}")
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
                        "best_f1/val": best_val_f1
                    })
                    
        with torch.no_grad():
            net.load_state_dict(torch.load(f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}/checkpoints/best_model.pt", map_location=primary_device))
            net.eval()
            test_results = val(net, test_loader, loss_fn, primary_device, args.tqdm_able)
            print("Test results:")
            print(test_results)
            os.makedirs("./results", exist_ok=True)
            with open(f'./results/{args.dataset}_{args.model}_{str(i_iter)}.txt','w') as f:    
                test_result_str = f'Accuracy:{test_results["acc"]}, Precision:{test_results["precision"]}, Recall:{test_results["recall"]}, F1:{test_results["f1"]}, Avg:{(test_results["acc"] + test_results["precision"]+ test_results["recall"]+ test_results["f1"])/4.0}'
                f.write(test_result_str)

            # 4. 【新增】将 Artifact 上传和当前迭代的测试指标记录移到循环内部，修复路径错误
            if args.if_wandb:
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
