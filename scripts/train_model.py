

from __future__ import annotations

import os
import json
import time
import glob
import random
import argparse
from typing import List, Sequence, Optional, Dict, Any

import numpy as np

# ensure project root is on sys.path so `models` package imports work when running this script directly
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Optional imports (deferred to runtime where possible).
# Import torch first; if tensorboard is missing we still want torch to be usable.
try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
except Exception:
    # torch isn't available in this environment; fill placeholders and we'll
    # raise a helpful error later when training is attempted.
    torch = None  # type: ignore
    nn = None  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = object  # type: ignore

# TensorBoard import is optional and should not cause the whole module import
# to fail if it's missing.
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except Exception:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

from tqdm import tqdm


# ----------------------------- Utilities ---------------------------------

def detect_device() -> torch.device:
    
    if torch is None:
        raise RuntimeError('PyTorch is required for training. Please install torch.')
    if torch.cuda.is_available():
        return torch.device('cuda')
    # MPS support (Apple Silicon) is detected via torch.backends
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def safe_load_npz(path: str) -> Dict[str, np.ndarray]:
    
    data = np.load(path)
    out: Dict[str, np.ndarray] = {}
    # prefer float32 for continuous volumes
    for k in data.files:
        arr = data[k]
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        out[k] = arr
    return out


# ----------------------------- Dataset -----------------------------------

class LazyTSDFDataset(Dataset):
    

    def __init__(self, paths: Sequence[str], channels: Sequence[str] = ('partial',),
                 dtype=np.float32, preload: bool = False):
        
        super().__init__()
        self.paths = list(paths)
        self.channels = list(channels)
        self.dtype = dtype
        self.preload = bool(preload)
        self._cache: Optional[List[Optional[Dict[str, np.ndarray]]]] = None
        if self.preload:
            # load everything immediately (useful for small datasets / debugging)
            self._cache = []
            for p in tqdm(self.paths, desc='Preloading NPZs'):
                self._cache.append(safe_load_npz(p))

    def __len__(self) -> int:
        return len(self.paths)

    def _load_item(self, idx: int) -> Dict[str, np.ndarray]:
        if self._cache is not None:
            data = self._cache[idx]
            if data is None:
                data = safe_load_npz(self.paths[idx])
                self._cache[idx] = data
            return data  # type: ignore
        return safe_load_npz(self.paths[idx])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self._load_item(idx)
        # target tsdf (required)
        if 'tsdf' not in data:
            raise KeyError(f"Missing 'tsdf' in {self.paths[idx]}")
        tsdf = data['tsdf'].astype(self.dtype)
        # ensure 3D shape
        if tsdf.ndim != 3:
            raise ValueError(f"tsdf must be 3D in {self.paths[idx]}, got shape {tsdf.shape}")

        # build inputs by concatenating requested channels
        input_chs: List[np.ndarray] = []
        for ch in self.channels:
            if ch == 'partial':
                arr = data.get('partial')
                if arr is None:
                    # If partial is missing, fall back to zeros
                    arr = np.zeros_like(tsdf, dtype=self.dtype)
                arr = arr.astype(self.dtype)
                input_chs.append(arr)
            elif ch == 'tsdf':
                # sometimes useful to provide the ground truth as an input (not recommended for training)
                input_chs.append(tsdf.astype(self.dtype))
            elif ch == 'occ' or ch == 'occupancy' or ch == 'occupancies':
                arr = data.get('occ') if 'occ' in data else data.get('occupancy')
                if arr is None:
                    arr = np.zeros_like(tsdf, dtype=np.uint8)
                # convert to float32 0/1
                arr = (arr.astype(np.float32) > 0).astype(self.dtype)
                input_chs.append(arr)
            elif ch == 'vis' or ch == 'visibility':
                arr = data.get('vis') or data.get('visibility')
                if arr is None:
                    arr = np.zeros_like(tsdf, dtype=np.uint8)
                arr = (arr.astype(np.float32) > 0).astype(self.dtype)
                input_chs.append(arr)
            elif ch == 'camera_dir' or ch == 'cam_dir':
                arr = data.get('camera_dir')
                if arr is None:
                    # if missing, provide zeros with 3 channels
                    arr = np.zeros((3,) + tsdf.shape, dtype=self.dtype)
                # expected shape (3,G,G,G)
                if arr.ndim == 3:
                    # maybe stored as single-channel direction magnitude? broadcast
                    arr = np.stack([arr, arr, arr], axis=0)
                if arr.shape[0] != 3:
                    # support (G,G,G,3) layout
                    if arr.ndim == 4 and arr.shape[-1] == 3:
                        arr = np.moveaxis(arr, -1, 0)
                    else:
                        raise ValueError(f'camera_dir has unsupported shape {arr.shape} in {self.paths[idx]}')
                # normalize to float
                arr = arr.astype(self.dtype)
                input_chs.append(arr)
            else:
                # Unknown channel name: try to fetch it directly
                arr = data.get(ch)
                if arr is None:
                    raise KeyError(f'Channel "{ch}" not found in {self.paths[idx]}')
                # if scalar, expand to volume
                if np.isscalar(arr):
                    arr = np.full_like(tsdf, float(arr), dtype=self.dtype)
                arr = arr.astype(self.dtype)
                # if arr is 3D -> single channel; if 4D with channel dim as first -> append each
                if arr.ndim == 3:
                    input_chs.append(arr)
                elif arr.ndim == 4:
                    # multiple leading channels
                    for c in range(arr.shape[0]):
                        input_chs.append(arr[c].astype(self.dtype))
                else:
                    raise ValueError(f'Unsupported array shape for channel {ch}: {arr.shape}')

        # stack channels into shape (C,G,G,G)
        if len(input_chs) == 0:
            # ensure there's at least one channel
            input_vol = np.zeros((1,) + tsdf.shape, dtype=self.dtype)
        else:
            # convert each to shape (1,G,G,G) if needed and stack
            nch = 0
            processed = []
            for a in input_chs:
                if a.ndim == 3:
                    processed.append(a[np.newaxis, ...])
                    nch += 1
                elif a.ndim == 4:
                    # already has channel dim first
                    processed.append(a)
                    nch += a.shape[0]
                else:
                    raise ValueError('unexpected array ndim')
            input_vol = np.concatenate(processed, axis=0)

        # create simple metadata for potential logging
        meta = {
            'path': self.paths[idx],
            'shape': tsdf.shape,
            'channels': self.channels,
        }

        # Convert to contiguous arrays to help PyTorch conversion
        input_vol = np.ascontiguousarray(input_vol)
        tsdf = np.ascontiguousarray(tsdf[np.newaxis, ...])  # (1,G,G,G)

        return {'inputs': input_vol, 'tsdf': tsdf, 'meta': meta}


def default_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    
    # stack inputs and tsdf
    inputs = np.stack([item['inputs'] for item in batch], axis=0)  # (B,C,G,G,G)
    tsdfs = np.stack([item['tsdf'] for item in batch], axis=0)     # (B,1,G,G,G)
    metas = [item.get('meta') for item in batch]
    return {'inputs': inputs, 'tsdf': tsdfs, 'meta': metas}


def worker_init_fn(worker_id: int, seed: int):
    
    # seed passed via closure in DataLoader call
    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    try:
        torch.manual_seed(worker_seed)
    except Exception:
        pass


# ----------------------------- Training ----------------------------------

def build_model(G: int, mode: str = 'small') -> nn.Module:
    
    try:
        from models.epn import get_model
    except Exception as e:
        raise RuntimeError('Failed to import models.epn.get_model: ' + str(e))

    # try the 'size' kwarg first
    try:
        return get_model(G, size=mode)
    except TypeError:
        # maybe the factory accepts only G or different signature
        try:
            return get_model(G)
        except Exception as e:
            raise RuntimeError('models.epn.get_model could not be called with G or size; error: ' + str(e))


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device, loss_fn: nn.Module, scaler: Optional[torch.cuda.amp.GradScaler],
                use_amp: bool) -> float:
    
    model.train()
    running_loss = 0.0
    n = 0
    pbar = tqdm(dataloader, desc='train', leave=False)
    for batch in pbar:
        # batch contains numpy arrays; move to torch tensors and device
        inputs = torch.from_numpy(batch['inputs']).to(device=device)
        targets = torch.from_numpy(batch['tsdf']).to(device=device)
        # expected shapes: inputs (B,C,G,G,G), targets (B,1,G,G,G)
        optimizer.zero_grad()
        if use_amp and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                preds = model(inputs)
                loss = loss_fn(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(inputs)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
        batch_loss = float(loss.item())
        running_loss += batch_loss * inputs.shape[0]
        n += inputs.shape[0]
        pbar.set_postfix({'loss': f'{(running_loss / n):.6f}'})
    avg = running_loss / max(1, n)
    return avg


def validate_epoch(model: nn.Module, dataloader: DataLoader, device: torch.device,
                   loss_fn: nn.Module, use_amp: bool) -> float:
    model.eval()
    running_loss = 0.0
    n = 0
    pbar = tqdm(dataloader, desc='val', leave=False)
    with torch.no_grad():
        for batch in pbar:
            inputs = torch.from_numpy(batch['inputs']).to(device=device)
            targets = torch.from_numpy(batch['tsdf']).to(device=device)
            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    preds = model(inputs)
                    loss = loss_fn(preds, targets)
            else:
                preds = model(inputs)
                loss = loss_fn(preds, targets)
            batch_loss = float(loss.item())
            running_loss += batch_loss * inputs.shape[0]
            n += inputs.shape[0]
            pbar.set_postfix({'val_loss': f'{(running_loss / n):.6f}'})
    avg = running_loss / max(1, n)
    return avg


# ------------------------------ CLI / Main --------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train 3D EPN-style models on TSDF .npz datasets')
    p.add_argument('--data-root', type=str, default='outputs',
                   help='Directory containing *_tsdf.npz files or a folder of class-specific npz files')
    p.add_argument('--pattern', type=str, default='*_tsdf.npz',
                   help='glob pattern for npz files under data-root')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--G', type=int, default=32, help='grid resolution (will be inferred if not provided)')
    p.add_argument('--model-mode', type=str, choices=['small', 'full', 'unet', 'conditioned'], default='small',
                   help='model size/mode to pass to get_model')
    p.add_argument('--channels', type=str, default='partial',
                   help='comma-separated channels to include as input (e.g. partial,vis,occ,camera_dir)')
    p.add_argument('--save-dir', type=str, default='outputs/checkpoints', help='where to save checkpoints and logs')
    p.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    p.add_argument('--val-split', type=float, default=0.1, help='fraction of data reserved for validation')
    p.add_argument('--num-workers', type=int, default=4, help='DataLoader num_workers')
    p.add_argument('--persistent-workers', action='store_true', help='enable DataLoader persistent_workers')
    p.add_argument('--mixed-precision', action='store_true', help='enable AMP (CUDA only)')
    p.add_argument('--save-every', type=int, default=1, help='checkpoint every N epochs (also saves best model)')
    p.add_argument('--pin-memory', action='store_true', help='enable pin_memory in DataLoader')
    p.add_argument('--preload', action='store_true', help='preload all npz into memory (for small datasets/debug)')
    p.add_argument('--no-tensorboard', action='store_true', help='disable tensorboard even if available')
    p.add_argument('--channels-list', type=str, default=None,
                   help=argparse.SUPPRESS)  # deprecated alias support
    return p.parse_args(argv)


def find_npz_files(root: str, pattern: str = '*_tsdf.npz') -> List[str]:
    """Find npz files under a root. If root is a single file, return it.
    Returns a sorted list of paths.
    """
    if os.path.isfile(root) and root.endswith('.npz'):
        return [root]
    pattern_path = os.path.join(root, pattern)
    files = sorted(glob.glob(pattern_path))
    return files


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # create save dir
    os.makedirs(args.save_dir, exist_ok=True)

    # prepare channel list
    channels = [c.strip() for c in args.channels.split(',') if c.strip()]
    if args.channels_list:
        # backward compat
        channels = [c.strip() for c in args.channels_list.split(',') if c.strip()]

    # find files
    npz_paths = find_npz_files(args.data_root, args.pattern)
    if len(npz_paths) == 0:
        print(f'No npz files found under {args.data_root} with pattern {args.pattern}')
        return

    # deterministic seed
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)

    # split train/val
    n = len(npz_paths)
    idxs = list(range(n))
    random.shuffle(idxs)
    n_val = max(1, int(n * args.val_split)) if args.val_split > 0 else 0
    val_idxs = set(idxs[:n_val])
    train_paths = [npz_paths[i] for i in idxs if i not in val_idxs]
    val_paths = [npz_paths[i] for i in idxs if i in val_idxs]

    print(f'Total samples: {n} (train={len(train_paths)}, val={len(val_paths)})')

    # infer G if not provided by loading the first sample header
    if args.G is None or args.G <= 0:
        sample = safe_load_npz(npz_paths[0])
        if 'tsdf' in sample:
            args.G = sample['tsdf'].shape[0]
        else:
            raise RuntimeError('Unable to infer G from sample; please pass --G')

    # build datasets
    train_ds = LazyTSDFDataset(train_paths, channels=channels, preload=args.preload)
    val_ds = LazyTSDFDataset(val_paths, channels=channels, preload=args.preload) if len(val_paths) > 0 else None

    # DataLoader settings
    use_persistent = bool(args.persistent_workers and args.num_workers > 0)
    collate = default_collate

    # worker_init_fn closure to include seed
    def _worker_init_fn(worker_id):
        worker_init_fn(worker_id, seed)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, persistent_workers=use_persistent,
                          pin_memory=args.pin_memory, collate_fn=collate, worker_init_fn=_worker_init_fn)
    val_dl = None
    if val_ds is not None:
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=max(0, args.num_workers // 2), persistent_workers=use_persistent,
                            pin_memory=args.pin_memory, collate_fn=collate, worker_init_fn=_worker_init_fn)

    # device
    device = detect_device()
    print('Using device:', device)

    # build model
    model = build_model(args.G, mode=args.model_mode)
    model = model.to(device)

    # optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # AMP setup
    use_amp = bool(args.mixed_precision and device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # TensorBoard
    tb_writer = None
    if TENSORBOARD_AVAILABLE and not args.no_tensorboard:
        try:
            tb_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'tb'))
        except Exception:
            tb_writer = None

    # checkpoint bookkeeping
    best_val = float('inf')
    best_path = os.path.join(args.save_dir, 'best.pth')
    last_path = os.path.join(args.save_dir, 'last.pth')
    # metrics CSV (append per-epoch results here)
    metrics_csv = os.path.join(args.save_dir, 'metrics_per_epoch.csv')
    # write header if not exists
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, 'w') as f:
            f.write('epoch,train_loss,val_loss,timestamp\n')

    # training loop
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f'=== Epoch {epoch}/{args.epochs} ===')
        train_loss = train_epoch(model, train_dl, optimizer, device, loss_fn, scaler, use_amp)
        print(f'Epoch {epoch} train loss: {train_loss:.6f}')
        if tb_writer:
            tb_writer.add_scalar('train/loss', train_loss, epoch)

        if val_dl is not None:
            val_loss = validate_epoch(model, val_dl, device, loss_fn, use_amp)
            print(f'Epoch {epoch} val loss: {val_loss:.6f}')
            if tb_writer:
                tb_writer.add_scalar('val/loss', val_loss, epoch)
        else:
            val_loss = None

        # checkpointing
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': vars(args),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'timestamp': time.time(),
        }
        # save last
        save_checkpoint(ckpt, last_path)

        # save best
        if (val_loss is not None and val_loss < best_val) or (val_loss is None and train_loss < best_val):
            best_val = val_loss if val_loss is not None else train_loss
            save_checkpoint(ckpt, best_path)
            print('Saved new best checkpoint to', best_path)

        # optionally save periodic checkpoint
        if epoch % args.save_every == 0:
            epoch_path = os.path.join(args.save_dir, f'epoch_{epoch}.pth')
            save_checkpoint(ckpt, epoch_path)
        # append epoch metrics to CSV (minimal logging)
        try:
            with open(metrics_csv, 'a') as f:
                f.write(f"{epoch},{train_loss},{val_loss if val_loss is not None else ''},{time.time()}\n")
        except Exception:
            pass

    total_time = time.time() - start_time
    print(f'Training finished in {total_time:.1f}s. Best val: {best_val:.6f}')
    if tb_writer:
        tb_writer.flush()
        tb_writer.close()


if __name__ == '__main__':
    main()
