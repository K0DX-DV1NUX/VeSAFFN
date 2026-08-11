

def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    base_lr = args.d_learning_rate
    schedule = args.d_lradj

    # ------------------------------------------------------------
    # constant
    # No learning rate decay. LR remains fixed for all epochs.
    # Useful for short experiments or when another mechanism
    # (e.g., early stopping) controls training.
    # ------------------------------------------------------------
    if schedule == "constant":
        lr = base_lr

    # ------------------------------------------------------------
    # type1
    # Aggressive exponential decay.
    # LR halves every epoch:
    #   lr = base_lr * (0.5)^(epoch-1)
    # Example:
    #   epoch 1 → base_lr
    #   epoch 2 → base_lr * 0.5
    #   epoch 3 → base_lr * 0.25
    # This schedule reduces LR very quickly and is typically
    # suitable only for short training runs.
    # ------------------------------------------------------------
    elif schedule == "type1":
        lr = base_lr * (0.5 ** ((epoch - 1) // 1))

    # ------------------------------------------------------------
    # type2
    # Moderately aggressive exponential decay after warmup.
    # First 3 epochs use base LR, then LR decays by factor 0.8
    # every epoch afterwards.
    #
    # Example:
    #   epoch 1–3 → base_lr
    #   epoch 4   → base_lr * 0.8
    #   epoch 5   → base_lr * 0.64
    #
    # This is smoother than type1 but still relatively fast
    # decay. Suitable for medium-length training runs.
    # ------------------------------------------------------------
    elif schedule == "type2":
        lr = base_lr if epoch < 3 else base_lr * (0.8 ** (epoch - 3))

    # ------------------------------------------------------------
    # type3 / type4 / type5 / type6
    # Step decay schedule (single drop).
    #
    # LR stays constant until a chosen epoch threshold,
    # then drops by a factor of 10 and stays constant again.
    #
    # thresholds:
    #   type3 → drop at epoch 5
    #   type4 → drop at epoch 10
    #   type5 → drop at epoch 15
    #   type6 → drop at epoch 20
    #
    # This is a common strategy for stable training and is
    # generally less aggressive than exponential decay.
    # ------------------------------------------------------------
    elif schedule in {"type3", "type4", "type5", "type6"}:
        thresholds = {"type3": 5, "type4": 10, "type5": 15, "type6": 20}
        lr = base_lr if epoch < thresholds[schedule] else base_lr * 0.1

    # ------------------------------------------------------------
    # type7
    # Slow and smooth step-wise decay.
    #
    # LR remains constant for the first 4 epochs, then decays
    # gradually by a factor of 0.7 every 4 epochs.
    #
    # Example:
    #   epoch 1–4  → base_lr
    #   epoch 5–8  → base_lr * 0.7
    #   epoch 9–12 → base_lr * 0.49
    #
    # A minimum LR floor (1e-10) prevents LR from collapsing
    # to extremely small values during long training runs.
    # This schedule is conservative and suitable for long runs.
    # ------------------------------------------------------------
    elif schedule == "type7":
        lr = base_lr if epoch <= 4 else base_lr * (0.7 ** ((epoch - 4) // 4))
        lr = max(lr, 1e-10)

    # ------------------------------------------------------------
    # Unknown schedule
    # ------------------------------------------------------------
    else:
        raise ValueError(f"Unknown lr schedule: {schedule}")

    # Apply updated LR to all optimizer parameter groups
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if printout:
        print(f"Updating learning rate to {lr}")