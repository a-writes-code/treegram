import os
import pickle

import torch
import torch.nn.functional as F


def tree_schedule(optimizer, init_lr, max_lr, frac_lr, warmup_steps, total_steps):
    fin_lr = frac_lr * max_lr

    def factor(step):
        if step < warmup_steps:
            return (init_lr + (max_lr - init_lr) * float(step / warmup_steps)) / init_lr
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return (0.5 * (1 + torch.cos(torch.tensor(torch.pi * progress))) * (max_lr - fin_lr) + fin_lr) / init_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def router_bias_schedule(lr, frac_lr, const_frac, total_steps):
    const_steps = int(const_frac * total_steps)
    return lambda step: lr if step < const_steps else frac_lr * lr


@torch.no_grad()
def validate(model, loader, device):
    was_training = model.training
    model.eval()
    loss_sum = torch.zeros((), device=device)
    count = torch.zeros((), device=device)
    for batch in loader:
        x = batch[:, :-1].contiguous().to(device)
        y = batch[:, 1:].contiguous().to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).float() * y.numel()
        count += float(y.numel())
    if was_training:
        model.train()
    return (loss_sum / count).item()


def train(model, optimizer, scheduler, train_loader, val_loader, device, epochs, update_freq,
          history_path, checkpoint_path, config, after_step=None):
    n_params = sum(p.numel() for p in model.parameters())
    total_steps = epochs * len(train_loader)
    update_freq = update_freq or max(total_steps // 100, 1)
    hist = {"train_loss": [], "val_loss": [], "steps": [], "tokens": [], "params": [n_params],
            "flops": [], "config": config}
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    steps = tokens = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            x = batch[:, :-1].contiguous().to(device)
            y = batch[:, 1:].contiguous().to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            tokens += y.numel()
            optimizer.step()
            scheduler.step()
            if after_step is not None:
                after_step(steps)

            if steps % update_freq == 0:
                val_loss = validate(model, val_loader, device)
                hist["train_loss"].append(loss.item())
                hist["val_loss"].append(val_loss)
                hist["steps"].append(steps)
                hist["tokens"].append(tokens)
                hist["flops"].append(6 * tokens * n_params)
                print(f"step {steps:6d}/{total_steps} | tokens {tokens:,} | train loss {loss.item():.4f} "
                      f"| val loss {val_loss:.4f}", flush=True)
                with open(history_path, "wb") as fp:
                    pickle.dump(hist, fp)

            optimizer.zero_grad(set_to_none=True)
            steps += 1

    with open(history_path, "wb") as fp:
        pickle.dump(hist, fp)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "step": steps, "tokens": tokens,
                "config": config}, checkpoint_path)
    return hist
