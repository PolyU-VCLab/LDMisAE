import math


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with multistep, cosine, or constant schedule after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        if args.lr_schedule == "constant":
            lr = args.lr
        elif args.lr_schedule == "cosine":
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        elif args.lr_schedule == "multistep":
            if epoch < 100:
                lr = 1e-4
            elif epoch < 200:
                lr = 1e-5
            elif epoch < 300:
                lr = 1e-6
            else:
                lr = 1e-7
        else:
            raise NotImplementedError(f"Unknown lr_schedule: {args.lr_schedule}")
    freeze_enc = args.enc_lr_scale_end_epoch > 0 and int(epoch) >= args.enc_lr_scale_end_epoch
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            if freeze_enc:
                param_group["lr"] = 0.0
            else:
                scale = param_group["lr_scale"]
                if args.enc_lr_scale_decay != 1.0:
                    decay_steps = int(epoch) // args.enc_lr_scale_decay_epochs
                    scale = scale * (args.enc_lr_scale_decay ** decay_steps)
                param_group["lr"] = lr * scale
        elif "dec_lr_scale" in param_group:
            scale = param_group["dec_lr_scale"]
            if args.dec_lr_scale_decay != 1.0:
                decay_steps = int(epoch) // args.dec_lr_scale_decay_epochs
                scale = scale * (args.dec_lr_scale_decay ** decay_steps)
            param_group["lr"] = lr * scale
        else:
            param_group["lr"] = lr
    return lr
