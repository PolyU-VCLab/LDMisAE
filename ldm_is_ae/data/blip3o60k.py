"""
BLIP3o-60K SFT dataset for JiT-half T2I refinement.
Pre-computed modulo sharding: all ranks scan all tars once at init,
each takes every Nth sample, then quota-trimmed so all ranks have strictly equal epoch lengths (total %% world_size != 0 would desync accum boundaries -> NCCL deadlock).
"""

import os, io, glob, random, signal, time, tarfile
from PIL import Image
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms


from ldm_is_ae.utils.crop import center_crop_arr


class _ReadTimeoutError(Exception):
    pass


class _ReadTimeout:
    def __init__(self, seconds=60):
        self.seconds = seconds

    def __enter__(self):
        self._old_handler = signal.signal(signal.SIGALRM, self._handler)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self._old_handler)
        return False

    @staticmethod
    def _handler(signum, frame):
        raise _ReadTimeoutError("Read timed out")


class BLIP3o60kDataset(IterableDataset):

    def __init__(
        self,
        tar_dir,
        image_size=256,
        shuffle=True,
        rank=0,
        world_size=1,
    ):
        super().__init__()
        self.tar_dir = tar_dir
        self.image_size = image_size
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size

        tar_files = sorted(glob.glob(os.path.join(tar_dir, '*.tar')))
        if not tar_files:
            raise FileNotFoundError(f"No .tar files found in {tar_dir}")

        # One full scan, identical and deterministic across all ranks: (tar, jpg_key, has_txt, global_idx)
        all_jpg = []
        global_idx = 0
        for tar_path in tar_files:
            with tarfile.open(tar_path) as tar:
                members = {m.name: m for m in tar.getmembers()}
                for jpg_key in sorted(k for k in members if k.endswith('.jpg')):
                    txt_key = jpg_key.rsplit('.', 1)[0] + '.txt'
                    all_jpg.append((tar_path, jpg_key, txt_key in members, global_idx))
                    global_idx += 1
        matched = [e for e in all_jpg if e[2]]
        # Strictly equal sharding: quota = the smallest matched count over the residues, so every rank has an identical __len__.
        counts = {}
        for e in matched:
            r = e[3] % world_size
            counts[r] = counts.get(r, 0) + 1
        quota = min(counts.values()) if counts else 0
        if quota <= 0:
            raise RuntimeError(f'BLIP3o60k: empty shard for rank {rank}')
        mine = [e for e in matched if e[3] % world_size == rank][:quota]
        entries_by_tar = {}
        for tar_path, jpg_key, _has, _idx in mine:
            entries_by_tar.setdefault(tar_path, []).append(jpg_key)

        self.entries_by_tar = entries_by_tar
        self._len = sum(len(v) for v in entries_by_tar.values())
        self._last_yield = None
        self._members_cache = {}
        print(f'[BLIP3o60k] rank={rank} matched_total={len(matched)} quota={quota} len={self._len}', flush=True)

    def __len__(self):
        return self._len

    def __iter__(self):
        # Worker-level tar sharding: with num_workers>0 each worker iterates only its own tar subset, and the
        # per-worker outputs add up to this rank's full quota (equal length across ranks is preserved)
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        nw = info.num_workers if info is not None else 1
        tar_items = list(self.entries_by_tar.items())
        if self.shuffle:
            random.shuffle(tar_items)
        if nw > 1:
            tar_items = tar_items[wid::nw]

        for tar_path, jpg_keys in tar_items:
            try:
                with tarfile.open(tar_path) as tar:
                    # Member-header cache: getmembers is very slow on the shared filesystem (~30-60 s per tar) and
                    # rescanning it every epoch used to be the main bottleneck; the cache is built lazily per worker
                    members = self._members_cache.get(tar_path)
                    if members is None:
                        members = {m.name: m for m in tar.getmembers()}
                        self._members_cache[tar_path] = members
                    active = [k for k in jpg_keys if k in members]
                    if self.shuffle:
                        random.shuffle(active)

                    for jpg_key in active:
                        txt_key = jpg_key.rsplit('.', 1)[0] + '.txt'
                        # Retry a failed read 3 times; on repeated failure reuse the previous sample -- silently dropping
                        # samples would make the per-rank counts differ and deadlock NCCL
                        sample = None
                        for _attempt in range(3):
                            try:
                                with _ReadTimeout(seconds=60):
                                    f_img = tar.extractfile(members[jpg_key])
                                    img_bytes = f_img.read()
                                    f_txt = tar.extractfile(members[txt_key])
                                    txt_bytes = f_txt.read()

                                pil_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                                caption = txt_bytes.decode('utf-8').strip()

                                pil_image = center_crop_arr(pil_image, self.image_size)
                                if self.shuffle and random.random() < 0.5:
                                    pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)

                                img_tensor = transforms.PILToTensor()(pil_image)
                                sample = (img_tensor, caption)
                                break
                            except _ReadTimeoutError:
                                time.sleep(2)
                            except Exception:
                                time.sleep(1)
                        if sample is None:
                            if self._last_yield is not None:
                                sample = self._last_yield
                                print(f'[BLIP3o60k] rank={self.rank} read failed after retry, reuse last sample', flush=True)
                            else:
                                continue
                        self._last_yield = sample
                        yield sample

            except Exception:
                continue
