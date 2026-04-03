import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from models.DepMamba import DepMamba


DEFAULT_CONFIG_PATH = Path("config/config.yaml")
DEFAULT_DATA_ROOT = Path("/datasets/work/hb-nhmrc-dhcp/work/liu275/Depression/DVLOG_Feature")
DEFAULT_CKPT = Path(
    "/datasets/work/hb-nhmrc-dhcp/work/liu275/Depression/DepMamba/mambamodels/"
    "dvlog_DepMamba_2/checkpoints/best_model.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-data ERF probe for DepMamba on D-Vlog")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT, help="D-Vlog feature root")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT, help="Path to pretrained weights")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config YAML")
    parser.add_argument(
        "--case_index",
        type=int,
        default=19,
        help="Global row index in labels.csv (recommended test rows: 19, 920, 14)",
    )
    parser.add_argument(
        "--probe",
        choices=["logit", "prepool_t0", "prepool_mid", "prepool_last"],
        default="logit",
        help="Probe target: final logit or pre-pooling hidden step",
    )
    parser.add_argument("--save_prefix", type=str, default="real_depmamba_erf", help="Output prefix")
    return parser.parse_args()


def load_model(model_cfg: dict, ckpt_path: Path, device: torch.device) -> DepMamba:
    model = DepMamba(**model_cfg).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_case_feature(data_root: Path, case_index: int):
    labels_path = data_root / "labels.csv"
    with labels_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if case_index < 0 or case_index >= len(rows):
        raise IndexError(f"case_index={case_index} out of range [0, {len(rows) - 1}]")

    row = rows[case_index]
    sample_id, label_text, _, gender, fold = row[:5]

    if fold != "test":
        print(f"[WARN] case_index={case_index} belongs to fold={fold}, not test.")

    video = np.load(data_root / sample_id / f"{sample_id}_visual.npy").astype(np.float32)
    audio = np.load(data_root / sample_id / f"{sample_id}_acoustic.npy").astype(np.float32)

    t = min(video.shape[0], audio.shape[0])
    x = np.concatenate([video[:t], audio[:t]], axis=1)

    meta = {
        "sample_id": sample_id,
        "label_text": label_text,
        "label": int(label_text == "depression"),
        "gender": gender,
        "fold": fold,
        "seq_len": t,
        "video_dim": video.shape[1],
        "audio_dim": audio.shape[1],
    }
    return x, meta


def select_target(model: DepMamba, x: torch.Tensor, mask: torch.Tensor, probe: str):
    if probe == "logit":
        logits = model(x, mask)
        target = logits[0, 0]
        return target, {"kind": "logit", "target_t": None}

    hook_data = {}

    def _hook(_, __, output):
        hook_data["pre_pool"] = output

    handle = model.enssm_encoder.register_forward_hook(_hook)
    logits = model(x, mask)
    _ = logits  # keep explicit forward in graph
    pre_pool = hook_data["pre_pool"]  # shape [1, T, D]
    t = pre_pool.shape[1]

    if probe == "prepool_t0":
        idx = 0
    elif probe == "prepool_mid":
        idx = t // 2
    else:
        idx = t - 1

    target = pre_pool[:, idx, :].sum()
    handle.remove()
    return target, {"kind": "prepool", "target_t": idx}


def run_real_erf_probe(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["mmmamba"]

    print(f"Loading model config from: {args.config}")
    print(f"Loading weights from: {args.checkpoint}")
    model = load_model(model_cfg, args.checkpoint, device)

    print(f"Loading real D-Vlog case from labels.csv index={args.case_index}")
    feature_np, meta = load_case_feature(args.data_root, args.case_index)
    print(f"Case meta: {meta}")

    x = torch.from_numpy(feature_np).unsqueeze(0).to(device)
    x.requires_grad_(True)
    mask = torch.ones(1, x.shape[1], device=device)

    target, target_meta = select_target(model, x, mask, args.probe)

    model.zero_grad(set_to_none=True)
    target.backward()

    grad = x.grad[0].detach().cpu().numpy()  # [T, 161]
    video_grad = grad[:, : meta["video_dim"]]
    audio_grad = grad[:, meta["video_dim"] :]

    erf_video = np.linalg.norm(video_grad, axis=1)
    erf_audio = np.linalg.norm(audio_grad, axis=1)
    erf_joint = np.linalg.norm(grad, axis=1)

    out_prefix = f"{args.save_prefix}_idx{args.case_index}_{args.probe}"
    np.savez(
        f"{out_prefix}.npz",
        erf_joint=erf_joint,
        erf_video=erf_video,
        erf_audio=erf_audio,
        meta=meta,
        target_meta=target_meta,
    )

    t_axis = np.arange(meta["seq_len"])
    plt.figure(figsize=(13, 6))
    plt.plot(t_axis, erf_joint, label="Joint (V+A)", linewidth=2.2, color="teal")
    plt.plot(t_axis, erf_video, label="Video", linewidth=1.4, alpha=0.9, color="royalblue")
    plt.plot(t_axis, erf_audio, label="Audio", linewidth=1.4, alpha=0.9, color="darkorange")
    plt.fill_between(t_axis, erf_joint, color="teal", alpha=0.15)

    if target_meta["target_t"] is not None:
        plt.axvline(target_meta["target_t"], linestyle=":", color="black", label=f"Probe t={target_meta['target_t']}")

    plt.title(
        "Real ERF of DepMamba on D-Vlog"
        f"\ncase_index={args.case_index}, sample_id={meta['sample_id']}, label={meta['label_text']}, len={meta['seq_len']}",
        fontsize=12,
    )
    plt.xlabel("Time Step t")
    plt.ylabel(r"Gradient Norm $||\partial y / \partial x_t||_F$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    fig_path = f"{out_prefix}.png"
    plt.savefig(fig_path, dpi=300)
    print(f"Probe finished. Saved figure: {fig_path}")
    print(f"Saved curve data: {out_prefix}.npz")


if __name__ == "__main__":
    run_real_erf_probe(parse_args())
