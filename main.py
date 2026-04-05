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

def train_epoch(net, train_loader, loss_fn, optimizer, device, current_epoch, total_epochs, tqdm_able):
    net.train()
    sample_count = 0
    running_loss = 0.
    correct_count = 0
    with tqdm(train_loader, desc=f"Training epoch {current_epoch}/{total_epochs}", leave=False, unit="batch", disable=tqdm_able) as pbar:
        for x, y, mask in pbar:
            x, y, mask = x.to(device), y.to(device).unsqueeze(1), mask.to(device)
            y_pred = net(x, mask)
            loss = loss_fn(y_pred, y.to(torch.float32))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            sample_count += x.shape[0]
            running_loss += loss.item() * x.shape[0]
            pred = (y_pred > 0.).int()
            correct_count += (pred == y).sum().item()
            pbar.set_postfix({"loss": running_loss / sample_count, "acc": correct_count / sample_count,})
    return {"loss": running_loss / sample_count, "acc": correct_count / sample_count,}

def val(net, val_loader, loss_fn, device, tqdm_able):
    net.eval()
    sample_count = 0
    running_loss = 0.
    TP, FP, TN, FN = 0, 0, 0, 0
    with torch.no_grad():
        with tqdm(val_loader, desc="Validating", leave=False, unit="batch", disable=tqdm_able) as pbar:
            for x, y, mask in pbar:
                x, y, mask = x.to(device), y.to(device).unsqueeze(1), mask.to(device)
                y_pred = net(x, mask)
                loss = loss_fn(y_pred, y.to(torch.float32))
                sample_count += x.shape[0]
                running_loss += loss.item() * x.shape[0]
                pred = (y_pred > 0.).int()
                TP += torch.sum((pred == 1) & (y == 1)).item()
                FP += torch.sum((pred == 1) & (y == 0)).item()
                TN += torch.sum((pred == 0) & (y == 0)).item()
                FN += torch.sum((pred == 0) & (y == 1)).item()
                l = running_loss / sample_count
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1_score = (2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0)
                accuracy = ((TP + TN) / sample_count if sample_count > 0 else 0.0)
                pbar.set_postfix({"loss": l, "acc": accuracy, "precision": precision, "recall": recall, "f1": f1_score,})
    l = running_loss / sample_count
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = (2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0)
    accuracy = ((TP + TN) / sample_count if sample_count > 0 else 0.0)
    return {"loss": l, "acc": accuracy, "precision": precision, "recall": recall, "f1": f1_score,}

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
        net = net.to(args.device[0])
        
        if len(args.device) > 1:
            net = torch.nn.DataParallel(net, device_ids=args.device)
            
        if args.dataset=='dvlog':
            train_loader = get_dvlog_dataloader(args.data_dir, "train", args.batch_size, args.train_gender)
            val_loader = get_dvlog_dataloader(args.data_dir, "valid", args.batch_size, args.test_gender)
            test_loader = get_dvlog_dataloader(args.data_dir, "test", args.batch_size, args.test_gender)
        elif args.dataset=='lmvd':
            train_loader = get_lmvd_dataloader(args.data_dir, "train", args.batch_size, args.train_gender)
            val_loader = get_lmvd_dataloader(args.data_dir, "valid", args.batch_size, args.test_gender)
            test_loader = get_lmvd_dataloader(args.data_dir, "test", args.batch_size, args.test_gender)
            
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
        best_val_acc = -1.0
        best_test_acc = -1.0
        
        if args.train:
            for epoch in range(args.epochs):
                train_results = train_epoch(net, train_loader, loss_fn, optimizer, args.device[0], epoch, args.epochs, args.tqdm_able)
                val_results = val(net, val_loader, loss_fn, args.device[0],args.tqdm_able)
                val_acc = (val_results["acc"] + val_results["precision"]+ val_results["recall"]+ val_results["f1"])/4.0
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(net.state_dict(),f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}/checkpoints/best_model.pt")
                if args.if_wandb:
                    wandb.log({"loss/train": train_results["loss"], "acc/train": train_results["acc"], "loss/val": val_results["loss"], "acc/val": val_results["acc"], "precision/val": val_results["precision"], "recall/val": val_results["recall"], "f1/val": val_results["f1"]})
                    
        with torch.no_grad():
            net.load_state_dict(torch.load(f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}/checkpoints/best_model.pt", map_location=args.device[0]))
            net.eval()
            test_results = val(net, test_loader, loss_fn, args.device[0],args.tqdm_able)
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