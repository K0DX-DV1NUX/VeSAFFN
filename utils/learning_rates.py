def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # Sets the LR for this epoch from d_lradj.
    base_lr = args.d_learning_rate
    schedule = args.d_lradj

    if schedule == "constant":
        lr = base_lr

    elif schedule == "type1":
        lr = base_lr * (0.5 ** ((epoch - 1) // 1))

    elif schedule == "type2":
        lr = base_lr if epoch < 3 else base_lr * (0.8 ** (epoch - 3))

    elif schedule in {"type3", "type4", "type5", "type6"}:
        thresholds = {"type3": 5, "type4": 10, "type5": 15, "type6": 20}
        lr = base_lr if epoch < thresholds[schedule] else base_lr * 0.1

    # Constant for 4 epochs, then x0.7 every 4 epochs, floor 1e-10.
    elif schedule == "type7":
        lr = base_lr if epoch <= 4 else base_lr * (0.7 ** ((epoch - 4) // 4))
        lr = max(lr, 1e-10)

    else:
        raise ValueError(f"Unknown lr schedule: {schedule}")

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if printout:
        print(f"Updating learning rate to {lr}")