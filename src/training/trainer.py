import torch
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, device, loss_fns):
    model.train()
    total_loss, spec_corr, sev_corr, chr_corr, total = 0, 0, 0, 0, 0
    spec_loss_fn, sev_loss_fn, chr_loss_fn = loss_fns

    for X, y_spec, y_sev, y_chr in tqdm(dataloader):
        X, y_spec, y_sev, y_chr = X.to(device), y_spec.to(device), y_sev.to(device), y_chr.to(device)
        out_spec, out_sev, out_chr = model(X)

        loss = spec_loss_fn(out_spec, y_spec) + sev_loss_fn(out_sev, y_sev) + chr_loss_fn(out_chr, y_chr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total += X.size(0)
        spec_corr += (out_spec.argmax(1) == y_spec).sum().item()
        sev_corr += (out_sev.argmax(1) == y_sev).sum().item()
        chr_corr += ((out_chr > 0.5).float() == y_chr).sum().item()

    return total_loss / total, spec_corr / total, sev_corr / total, chr_corr / total

def validate(model, dataloader, device, loss_fns):
    model.eval()
    total_loss, spec_corr, sev_corr, chr_corr, total = 0, 0, 0, 0, 0
    spec_loss_fn, sev_loss_fn, chr_loss_fn = loss_fns

    y_spec_true, y_spec_pred = [], []
    y_sev_true, y_sev_pred = [], []
    y_chr_true, y_chr_pred = [], []

    with torch.no_grad():
        for X, y_spec, y_sev, y_chr in dataloader:
            X, y_spec, y_sev, y_chr = X.to(device), y_spec.to(device), y_sev.to(device), y_chr.to(device)
            out_spec, out_sev, out_chr = model(X)

            loss = spec_loss_fn(out_spec, y_spec) + sev_loss_fn(out_sev, y_sev) + chr_loss_fn(out_chr, y_chr)
            total_loss += loss.item() * X.size(0)
            total += X.size(0)

            pred_spec = out_spec.argmax(1)
            pred_sev = out_sev.argmax(1)
            pred_chr = (out_chr > 0.5).float()

            spec_corr += (pred_spec == y_spec).sum().item()
            sev_corr += (pred_sev == y_sev).sum().item()
            chr_corr += (pred_chr == y_chr).sum().item()

            y_spec_true.extend(y_spec.cpu().numpy())
            y_spec_pred.extend(pred_spec.cpu().numpy())
            y_sev_true.extend(y_sev.cpu().numpy())
            y_sev_pred.extend(pred_sev.cpu().numpy())
            y_chr_true.extend(y_chr.cpu().numpy())
            y_chr_pred.extend(pred_chr.cpu().numpy())

    return (total_loss / total, spec_corr / total, sev_corr / total, chr_corr / total,
            y_spec_true, y_spec_pred, y_sev_true, y_sev_pred, y_chr_true, y_chr_pred)
