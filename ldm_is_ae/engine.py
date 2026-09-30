import math
import sys
import os
import contextlib
from pathlib import Path

import torch
import numpy as np
import cv2

import ldm_is_ae.utils.misc as misc
import ldm_is_ae.utils.lr_sched as lr_sched
import copy
from ldm_is_ae.utils.train_log import logger
from ldm_is_ae.denoiser import rgb_full_to_grid

def _amp_train_ctx(args):
    if getattr(args, "fp32", False):
        return contextlib.nullcontext()
    return torch.amp.autocast("cuda", dtype=torch.bfloat16)


def _eval_generate_ctx(args, device):
    """Multi-step sampling diverges numerically under bf16 autocast (garbled images); generate() runs in FP32 for inference and online eval."""
    if getattr(args, "sample_bf16", False) and device.type == "cuda" and not getattr(args, "fp32", False):
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _eval_swap_params_to_ema1(model_without_ddp):
    ema_list = getattr(model_without_ddp, "ema_params1", None)
    if ema_list is None:
        raise RuntimeError("ema_params1 is missing; load a checkpoint with model_ema1")
    named = list(model_without_ddp.named_parameters())
    if len(ema_list) != len(named):
        raise RuntimeError("ema_params1 len {} != named_parameters {}".format(len(ema_list), len(named)))
    train_backup = [p.detach().clone() for (_, p) in named]
    for (name, p), ema in zip(named, ema_list):
        e = ema.to(device=p.device, dtype=p.dtype, non_blocking=True)
        if e.shape != p.shape:
            raise RuntimeError("EMA vs param shape mismatch {}: {} {}".format(name, tuple(e.shape), tuple(p.shape)))
        p.data.copy_(e)
    return train_backup


def _eval_restore_train_params(model_without_ddp, train_backup):
    for (_, p), b in zip(model_without_ddp.named_parameters(), train_backup):
        p.data.copy_(b.to(device=p.device, dtype=p.dtype, non_blocking=True))


def _raw_norm_scale_value(denoiser):
    net = getattr(denoiser, "net", None)
    net = getattr(net, "_orig_mod", net)
    raw_norm = getattr(net, "raw_norm", None)
    if raw_norm is None or not hasattr(raw_norm, "scale_logit"):
        return float("nan")
    with torch.no_grad():
        return (raw_norm.max_scale * torch.sigmoid(raw_norm.scale_logit.detach().float())).mean().item()


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None, max_optimizer_steps=None, text_encoder=None, global_step_start=0):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    accum_iter = max(1, int(getattr(args, 'accum_iter', 1)))
    num_steps = len(data_loader)
    optimizer.zero_grad()
    opt_steps_done = 0

    if log_writer is not None:
        logger.info('log_dir: {}', log_writer.log_dir)

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / num_steps + epoch, args)

        step_1 = data_iter_step + 1
        remainder = step_1 % accum_iter
        at_accum_boundary = (remainder == 0) or (step_1 == num_steps)
        if step_1 == num_steps and remainder != 0:
            accum_scale = remainder
        else:
            accum_scale = accum_iter

        sync_cm = contextlib.nullcontext() if (accum_iter <= 1 or at_accum_boundary) else model.no_sync()

        with sync_cm:
            x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
            x = x * 2.0 - 1.0
            x = rgb_full_to_grid(x, args.pixel_patch_size)
            if args.text_dim > 0 and text_encoder is not None:
                labels = text_encoder(list(labels)).to(device, non_blocking=True)
            else:
                labels = labels.to(device, non_blocking=True)
            with _amp_train_ctx(args):
                loss_dict = model(x, labels)
            loss = loss_dict["loss"]
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                logger.error("Loss is {}, stopping training", loss_value)
                sys.exit(1)
            (loss / accum_scale).backward()
            if misc.is_main_process() and (data_iter_step % print_freq == 0 or data_iter_step == num_steps - 1):
                logger.info(
                    "loss={:.6f} x_mse={:.6f} v_mse={:.6f} v_cohesion={:.6f}, z_mse={:.6f} lpips={:.6f} raw_norm_scale={:.6f}",
                    loss_dict["loss"].float().item(),
                    loss_dict["x_mse"].float().item(),
                    loss_dict["v_mse"].float().item(),
                    loss_dict["v_cohesion"].float().item(),
                    loss_dict["z_mse"].float().item(),
                    loss_dict["lpips"].float().item(),
                    _raw_norm_scale_value(model_without_ddp),
                )

        if at_accum_boundary:
            cgn = float(getattr(args, "clip_grad_norm", 0.0) or 0.0)
            if cgn > 0.0:
                # Numerically equivalent to the classic per-tensor path (same total norm, different implementation route)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cgn, foreach=False)
            optimizer.step()
            model_without_ddp.update_ema()
            optimizer.zero_grad()
            opt_steps_done += 1
            global_step = global_step_start + opt_steps_done

            save_every = int(getattr(args, "save_every_n_steps", 0) or 0)
            if save_every > 0 and global_step % save_every == 0:
                misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                epoch=epoch, epoch_name=str(global_step), global_step=global_step)
                misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                epoch=epoch, epoch_name="last", global_step=global_step)

            # Per-step online eval (output dir: eval_{global_step}_t2i)
            eval_every = int(getattr(args, "eval_every_n_steps", 0) or 0)
            if args.online_eval and eval_every > 0 and global_step % eval_every == 0:
                torch.cuda.empty_cache()
                with torch.no_grad():
                    evaluate(model_without_ddp, args, global_step, batch_size=args.gen_bsz,
                             log_writer=log_writer, text_encoder=text_encoder)
                torch.cuda.empty_cache()

            if max_optimizer_steps is not None and opt_steps_done >= max_optimizer_steps:
                break

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)

    return opt_steps_done


def _evaluate_t2i(model_without_ddp, args, epoch, batch_size=64, log_writer=None, text_encoder=None):
    """T2I online eval: generate images batch by batch from --eval_prompts (text-token conditioning)."""
    epoch_str = str(epoch)
    save_folder = os.path.join(args.output_dir, "eval_{}_t2i".format(epoch_str))
    if log_writer is not None:
        logger.info("log_dir: {}", log_writer.log_dir)
    logger.info("Switch to ema")
    train_backup = _eval_swap_params_to_ema1(model_without_ddp)
    mod_dev = next(model_without_ddp.parameters()).device

    prompt_path = getattr(args, "eval_prompts", "")
    if not prompt_path or not os.path.isfile(prompt_path):
        logger.warning("eval_prompts not found: {}; skipping T2I eval", prompt_path)
        _eval_restore_train_params(model_without_ddp, train_backup)
        return
    if text_encoder is None:
        logger.warning("text_encoder is None; skipping T2I generation")
        _eval_restore_train_params(model_without_ddp, train_backup)
        return

    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    total = min(len(prompts), args.num_images) if args.num_images > 0 else len(prompts)
    prompts = prompts[:total]
    logger.info("T2I eval: {} prompts -> {}", total, save_folder)
    os.makedirs(save_folder, exist_ok=True)

    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    gen_bs = max(1, min(int(batch_size), total))
    num_groups = (total + gen_bs - 1) // gen_bs
    base_start_idx = np.arange(num_groups, dtype=np.int64) * gen_bs
    start_idx_list = [int(base_start_idx[i]) for i in range(local_rank, num_groups, world_size)]

    for step_i, start_idx in enumerate(start_idx_list):
        end_idx = min(start_idx + gen_bs, total)
        g = start_idx // gen_bs
        lo, hi = int(start_idx), int(end_idx)
        batch_done = lo >= hi or all(
            os.path.isfile(os.path.join(save_folder, "{}.png".format(str(i).zfill(5)))) for i in range(lo, hi)
        )
        if batch_done:
            logger.info("skip: rank={} step={}/{} img_id=[{}, {})", local_rank, step_i + 1, len(start_idx_list), lo, hi)
            continue
        batch_prompts = list(prompts[start_idx:end_idx])
        token_seqs = text_encoder(batch_prompts).to(mod_dev, non_blocking=True)
        gen_seed = int(start_idx + int(getattr(args, "seed", 0)))
        torch.manual_seed(gen_seed)
        torch.cuda.manual_seed_all(gen_seed)
        with _eval_generate_ctx(args, mod_dev):
            sampled_images = model_without_ddp.generate(token_seqs)
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()
        for b_id in range(sampled_images.size(0)):
            img_id = g * gen_bs + b_id
            if img_id >= total:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].float().numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, "{}.png".format(str(img_id).zfill(5))), gen_img)

    torch.distributed.barrier()
    logger.info("Switch back from ema")
    _eval_restore_train_params(model_without_ddp, train_backup)


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None, text_encoder=None):

    if args.text_dim > 0:
        _evaluate_t2i(model_without_ddp, args, epoch, batch_size=batch_size, log_writer=log_writer, text_encoder=text_encoder)
        return

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    gen_bs = int(batch_size)
    num_groups = (args.num_images + gen_bs - 1) // gen_bs

    save_folder = getattr(args, "gen_images_dir", None) or os.path.join(
        args.output_dir,
        "epoch{:04d}-{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            epoch, model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    logger.info("Save to: {}", save_folder)
    os.makedirs(save_folder, exist_ok=True)

    # Align the EMA in place with param.data.copy_ instead of load_state_dict on a deepcopy-built dict, which is sensitive to buffer / key ordering
    ema_mode = getattr(args, "ema_mode", "ema1")
    if ema_mode != "none":
        logger.info("Switch to ema")
        train_backup = _eval_swap_params_to_ema1(model_without_ddp)
    else:
        train_backup = None
    mod_dev = next(model_without_ddp.parameters()).device

    assert args.num_images > 0 and args.class_num > 0
    total_slots = num_groups * gen_bs
    class_label_gen_world = np.zeros(total_slots, dtype=np.int64)
    class_label_gen_world[:args.num_images] = (np.arange(args.num_images, dtype=np.int64) % int(args.class_num))

    base_start_idx = np.arange(num_groups, dtype=np.int64) * gen_bs
    if getattr(args, "random_idx", False):
        buf = torch.zeros(num_groups, dtype=torch.int64, device=mod_dev)
        if misc.is_main_process():
            ephem = int.from_bytes(os.urandom(8), "little")
            rng = np.random.default_rng(ephem)
            shuffled = base_start_idx[rng.permutation(num_groups)]
            buf.copy_(torch.as_tensor(shuffled, dtype=torch.int64, device=mod_dev))
            logger.info("random_idx: shuffle on main; ephemeral_shuffle_seed={}", ephem)
        if misc.is_dist_avail_and_initialized() and world_size > 1:
            torch.distributed.broadcast(buf, src=0)
        all_start_idx = buf.detach().cpu().numpy()
    else:
        all_start_idx = base_start_idx
    start_idx_list = [int(all_start_idx[i]) for i in range(local_rank, num_groups, world_size)]

    for step_i, start_idx in enumerate(start_idx_list):
        logger.info("Generation step {}/{}", step_i + 1, len(start_idx_list))
        g = start_idx // gen_bs
        end_idx = start_idx + gen_bs
        lo, hi = int(start_idx), min(int(end_idx), int(args.num_images))
        batch_done = lo >= hi or all(
            os.path.isfile(os.path.join(save_folder, "{}.png".format(str(img_id).zfill(5)))) for img_id in range(lo, hi)
        )
        if batch_done:
            logger.info(
                "skip: rank={} step={}/{} start_idx={} img_id=[{}, {}) {}",
                local_rank, step_i + 1, len(start_idx_list), start_idx, lo, hi,
                "(no images)" if lo >= hi else "(all png exist)",
            )
            continue
        labels_gen = torch.as_tensor(class_label_gen_world[start_idx:end_idx], dtype=torch.long, device=mod_dev).contiguous()
        global_seed = int(getattr(args, "seed", 0))
        gen_seed = int(start_idx + global_seed)
        torch.manual_seed(gen_seed)
        torch.cuda.manual_seed_all(gen_seed)
        with _eval_generate_ctx(args, mod_dev):
            sampled_images = model_without_ddp.generate(labels_gen)
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()
        for b_id in range(sampled_images.size(0)):
            img_id = g * gen_bs + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].float().numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    torch.distributed.barrier()

    if train_backup is not None:
        logger.info("Switch back from ema")
        _eval_restore_train_params(model_without_ddp, train_backup)


    torch.distributed.barrier()
