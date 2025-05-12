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

    with torch.no_grad():
        for X, y_spec, y_sev, y_chr in dataloader:
            X, y_spec, y_sev, y_chr = X.to(device), y_spec.to(device), y_sev.to(device), y_chr.to(device)
            out_spec, out_sev, out_chr = model(X)

            loss = spec_loss_fn(out_spec, y_spec) + sev_loss_fn(out_sev, y_sev) + chr_loss_fn(out_chr, y_chr)
            total_loss += loss.item() * X.size(0)
            total += X.size(0)

            spec_corr += (out_spec.argmax(1) == y_spec).sum().item()
            sev_corr += (out_sev.argmax(1) == y_sev).sum().item()
            chr_corr += ((out_chr > 0.5).float() == y_chr).sum().item()

    return total_loss / total, spec_corr / total, sev_corr / total, chr_corr / total

def train_and_evaluate(model, train_loader, val_loader, optimizer, loss_fns, n_epochs, early_stopping_patience, model_path, device):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")

        # Training
        train_loss, train_spec_acc, train_sev_acc, train_chr_acc = train_epoch(
            model, train_loader, optimizer, device, loss_fns
        )

        # Validation
        val_loss, val_spec_acc, val_sev_acc, val_chr_acc = validate(
            model, val_loader, device, loss_fns
        )

        print(f"Train Loss: {train_loss:.4f}, Spec Acc: {train_spec_acc:.4f}, Sev Acc: {train_sev_acc:.4f}, Chr Acc: {train_chr_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Spec Acc: {val_spec_acc:.4f}, Sev Acc: {val_sev_acc:.4f}, Chr Acc: {val_chr_acc:.4f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the model
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{early_stopping_patience}")

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break
