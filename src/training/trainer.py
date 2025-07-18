import torch
from tqdm import tqdm
import mlflow

def train_epoch(model, dataloader, optimizer, device, loss_fns):
    model.train()
    total_loss, spec_corr, sev_corr, chr_corr, total = 0, 0, 0, 0, 0
    spec_loss_fn, sev_loss_fn, chr_loss_fn = loss_fns

    for X, y_spec, y_sev, y_chr in dataloader:
        X, y_spec, y_sev, y_chr = X.to(device), y_spec.to(device), y_sev.to(device), y_chr.to(device)
        y_chr = y_chr.unsqueeze(1)

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
            y_chr = y_chr.unsqueeze(1)

            out_spec, out_sev, out_chr = model(X)
            loss = spec_loss_fn(out_spec, y_spec) + sev_loss_fn(out_sev, y_sev) + chr_loss_fn(out_chr, y_chr)

            total_loss += loss.item() * X.size(0)
            total += X.size(0)
            spec_corr += (out_spec.argmax(1) == y_spec).sum().item()
            sev_corr += (out_sev.argmax(1) == y_sev).sum().item()
            chr_corr += ((out_chr > 0.5).float() == y_chr).sum().item()

    return total_loss / total, spec_corr / total, sev_corr / total, chr_corr / total

def train_and_evaluate(
    model, train_loader, val_loader, optimizer, loss_fns, n_epochs, early_stopping_patience, model_path, device, run_id
):
    #To keep track of the best validation loss
    best_val_loss = float('inf')
    #Counter used for early stopping
    patience_counter = 0

    #If the loss does not improve for this 3 consecutives epochs, learning rate will be reduced of 50%
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)


    # Outer loop for epochs
    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}/{n_epochs}")

        # Training

        #Line for visualization of the training progress
        train_pbar = tqdm(train_loader, desc=f"Training - Epoch {epoch}/{n_epochs}", leave=True)

        train_loss, train_spec_acc, train_sev_acc, train_chr_acc = train_epoch(
            model, train_pbar, optimizer, device, loss_fns
        )

        # Validation
        val_loss, val_spec_acc, val_sev_acc, val_chr_acc = validate(
            model, val_loader, device, loss_fns
        )
        
        #Comunicate the validation loss to the scheduler
        scheduler.step(val_loss)

        # Logging metrics to MLflow
        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_spec_acc", train_spec_acc, step=epoch)
            mlflow.log_metric("train_sev_acc", train_sev_acc, step=epoch)
            mlflow.log_metric("train_chr_acc", train_chr_acc, step=epoch)

            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_spec_acc", val_spec_acc, step=epoch)
            mlflow.log_metric("val_sev_acc", val_sev_acc, step=epoch)
            mlflow.log_metric("val_chr_acc", val_chr_acc, step=epoch)

        print(f"Train Loss: {train_loss:.4f}, Spec Acc: {train_spec_acc:.4f}, Sev Acc: {train_sev_acc:.4f}, Chr Acc: {train_chr_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Spec Acc: {val_spec_acc:.4f}, Sev Acc: {val_sev_acc:.4f}, Chr Acc: {val_chr_acc:.4f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model hyperparameters
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

            # Log model to MLflow
            with mlflow.start_run(run_id=run_id, nested=True):
                mlflow.pytorch.log_model(model, "best_model")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break
