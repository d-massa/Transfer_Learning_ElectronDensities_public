import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score as sklearn_r2, mean_squared_error
import numpy as np
import os
from clip_cust import CLIPRegressionModel, r2_score
from torch.utils.tensorboard import SummaryWriter

def experiment(suffix):
    # Check if the directory exists
    if not os.path.exists('./outs'+suffix):
        # If it does not exist, create the directory
        os.makedirs('./outs'+suffix)
        print(f"Directory 'outs{suffix}' created.")
    else:
        print(f"Directory 'outs{suffix}' already exists.")
              
def kfold_train(epochs,k_folds,seedd,train_val_dataset,batch_size,clip_model,learning_rate,device,suffix,patience):

    best_model = None
    log_dir='./outs'+suffix

    # K-Fold Cross-Validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seedd)
    fold_results_r2 = []
    fold_results_rmse = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset)):
        print(f'Fold {fold + 1}/{k_folds}')

        # Create data loaders for this fold
        train_subset = Subset(train_val_dataset, train_idx)
        val_subset = Subset(train_val_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Model, Loss, Optimizer
        model = CLIPRegressionModel(clip_model).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        fold_log_dir = os.path.join(log_dir, f'fold_{fold + 1}')
        writer = SummaryWriter(log_dir=fold_log_dir)

        # Training loop
        best_r2 = -float('inf')
        best_rmse = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            running_r2 = 0.0
            running_rmse = 0.0
            for batch in train_loader:
                inputs, targets = batch
                pixel_values = inputs['pixel_values'].squeeze(1).to(device)
                input_ids = inputs['input_ids'].squeeze(1).to(device)
                attention_mask = inputs['attention_mask'].squeeze(1).to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(pixel_values, input_ids, attention_mask)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_r2 += r2_score(targets, outputs)
                running_rmse += np.sqrt(mean_squared_error(targets.cpu().numpy(), outputs.cpu().detach().numpy()))

            avg_loss = running_loss / len(train_loader)
            avg_r2 = running_r2 / len(train_loader)
            avg_rmse = running_rmse / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, R^2: {avg_r2:.4f}, RMSE: {avg_rmse:.4f}")

            # Log training metrics to TensorBoard
            writer.add_scalar('Loss/train', avg_loss, epoch)
            writer.add_scalar('R2/train', avg_r2, epoch)
            writer.add_scalar('RMSE/train', avg_rmse, epoch)

        # Validation loop
        model.eval()
        val_targets = []
        val_predictions = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                pixel_values = inputs['pixel_values'].squeeze(1).to(device)
                input_ids = inputs['input_ids'].squeeze(1).to(device)
                attention_mask = inputs['attention_mask'].squeeze(1).to(device)
                targets = targets.to(device)

                outputs = model(pixel_values, input_ids, attention_mask)

                val_targets.append(targets.cpu().numpy())
                val_predictions.append(outputs.cpu().numpy())

        val_targets = np.concatenate(val_targets, axis=0)
        val_predictions = np.concatenate(val_predictions, axis=0)

        r2 = sklearn_r2(val_targets, val_predictions)
        rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
        
        fold_results_r2.append(r2)
        fold_results_rmse.append(rmse)
        print(f"Fold {fold + 1}, Epoch {epoch+1}, Validation R^2: {r2:.4f}, Validation RMSE: {rmse:.4f}")


        writer.add_scalar('R2/validation', r2, fold)
        writer.add_scalar('RMSE/validation', rmse, epoch)

        # Check for early stopping
        if r2 > best_r2:
            best_r2 = r2
            epochs_no_improve = 0
            best_model = model
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        writer.close()

    # Average R^2 across folds
    mean_r2 = np.mean(fold_results_r2)
    mean_rmse = np.mean(fold_results_rmse)

    print(f"Mean R^2 across folds: {mean_r2:.4f}")
    print(f"Mean rmse across folds: {mean_rmse:.4f}")
    if best_model is not None:
        torch.save(best_model.state_dict(), f'outs{suffix}/best_model{suffix}.pth')

    return mean_r2,mean_rmse

def kfold_train_val_monit_r2(epochs, k_folds, seedd, train_val_dataset, batch_size, clip_model, learning_rate, device, suffix, patience):

    best_model = None
    log_dir = './outs' + suffix

    # K-Fold Cross-Validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seedd)
    fold_results_r2 = []
    fold_results_rmse = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset)):
        print(f'Fold {fold + 1}/{k_folds}')

        # Create data loaders for this fold
        train_subset = Subset(train_val_dataset, train_idx)
        val_subset = Subset(train_val_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Model, Loss, Optimizer
        model = CLIPRegressionModel(clip_model).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        fold_log_dir = os.path.join(log_dir, f'fold_{fold + 1}')
        writer = SummaryWriter(log_dir=fold_log_dir)

        # Training loop with validation monitoring
        best_r2 = -float('inf')
        best_rmse = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            running_r2 = 0.0
            running_rmse = 0.0
            for batch in train_loader:
                inputs, targets = batch
                pixel_values = inputs['pixel_values'].squeeze(1).to(device)
                input_ids = inputs['input_ids'].squeeze(1).to(device)
                attention_mask = inputs['attention_mask'].squeeze(1).to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(pixel_values, input_ids, attention_mask)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_r2 += r2_score(targets, outputs)
                running_rmse += np.sqrt(mean_squared_error(targets.cpu().numpy(), outputs.cpu().detach().numpy()))

            avg_loss = running_loss / len(train_loader)
            avg_r2 = running_r2 / len(train_loader)
            avg_rmse = running_rmse / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, R^2: {avg_r2:.4f}, RMSE: {avg_rmse:.4f}")

            # Log training metrics to TensorBoard
            writer.add_scalar('Loss/train', avg_loss, epoch)
            writer.add_scalar('R2/train', avg_r2, epoch)
            writer.add_scalar('RMSE/train', avg_rmse, epoch)

            # Validation loop to monitor performance after each epoch
            model.eval()
            val_targets = []
            val_predictions = []
            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = batch
                    pixel_values = inputs['pixel_values'].squeeze(1).to(device)
                    input_ids = inputs['input_ids'].squeeze(1).to(device)
                    attention_mask = inputs['attention_mask'].squeeze(1).to(device)
                    targets = targets.to(device)

                    outputs = model(pixel_values, input_ids, attention_mask)

                    val_targets.append(targets.cpu().numpy())
                    val_predictions.append(outputs.cpu().numpy())

            val_targets = np.concatenate(val_targets, axis=0)
            val_predictions = np.concatenate(val_predictions, axis=0)

            r2 = sklearn_r2(val_targets, val_predictions)
            rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))

            # Log validation metrics to TensorBoard
            writer.add_scalar('R2/validation', r2, epoch)
            writer.add_scalar('RMSE/validation', rmse, epoch)

            print(f"Fold {fold + 1}, Epoch {epoch + 1}, Validation R^2: {r2:.4f}, Validation RMSE: {rmse:.4f}")

            # Early stopping mechanism
            if r2 > best_r2:
                best_r2 = r2
                epochs_no_improve = 0
                best_model = model
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        writer.close()

        fold_results_r2.append(r2)
        fold_results_rmse.append(rmse)

    # Average R^2 across folds
    mean_r2 = np.mean(fold_results_r2)
    mean_rmse = np.mean(fold_results_rmse)

    print(f"Mean R^2 across folds: {mean_r2:.4f}")
    print(f"Mean RMSE across folds: {mean_rmse:.4f}")
    if best_model is not None:
        torch.save(best_model.state_dict(), f'outs{suffix}/best_model{suffix}.pth')

    return mean_r2, mean_rmse

def kfold_train_val_monit(epochs, k_folds, seedd, train_val_dataset, batch_size, clip_model, learning_rate, device, suffix, patience):

    best_model = None
    log_dir = './outs' + suffix

    # K-Fold Cross-Validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seedd)
    fold_results_r2 = []
    fold_results_rmse = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset)):
        print(f'Fold {fold + 1}/{k_folds}')

        # Create data loaders for this fold
        train_subset = Subset(train_val_dataset, train_idx)
        val_subset = Subset(train_val_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Model, Loss, Optimizer
        model = CLIPRegressionModel(clip_model).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        fold_log_dir = os.path.join(log_dir, f'fold_{fold + 1}')
        writer = SummaryWriter(log_dir=fold_log_dir)

        # Training loop with validation monitoring
        best_rmse = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            running_r2 = 0.0
            running_rmse = 0.0
            for batch in train_loader:
                inputs, targets = batch
                pixel_values = inputs['pixel_values'].squeeze(1).to(device)
                input_ids = inputs['input_ids'].squeeze(1).to(device)
                attention_mask = inputs['attention_mask'].squeeze(1).to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(pixel_values, input_ids, attention_mask)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_r2 += r2_score(targets, outputs)
                running_rmse += np.sqrt(mean_squared_error(targets.cpu().numpy(), outputs.cpu().detach().numpy()))

            avg_loss = running_loss / len(train_loader)
            avg_r2 = running_r2 / len(train_loader)
            avg_rmse = running_rmse / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, R^2: {avg_r2:.4f}, RMSE: {avg_rmse:.4f}")

            # Log training metrics to TensorBoard
            writer.add_scalar('Loss/train', avg_loss, epoch)
            writer.add_scalar('R2/train', avg_r2, epoch)
            writer.add_scalar('RMSE/train', avg_rmse, epoch)

            # Validation loop to monitor performance after each epoch
            model.eval()
            val_targets = []
            val_predictions = []
            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = batch
                    pixel_values = inputs['pixel_values'].squeeze(1).to(device)
                    input_ids = inputs['input_ids'].squeeze(1).to(device)
                    attention_mask = inputs['attention_mask'].squeeze(1).to(device)
                    targets = targets.to(device)

                    outputs = model(pixel_values, input_ids, attention_mask)

                    val_targets.append(targets.cpu().numpy())
                    val_predictions.append(outputs.cpu().numpy())

            val_targets = np.concatenate(val_targets, axis=0)
            val_predictions = np.concatenate(val_predictions, axis=0)

            r2 = sklearn_r2(val_targets, val_predictions)
            rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))

            # Log validation metrics to TensorBoard
            writer.add_scalar('R2/validation', r2, epoch)
            writer.add_scalar('RMSE/validation', rmse, epoch)

            print(f"Fold {fold + 1}, Epoch {epoch + 1}, Validation R^2: {r2:.4f}, Validation RMSE: {rmse:.4f}")

            # Early stopping mechanism based on RMSE
            if rmse < best_rmse:
                best_rmse = rmse
                epochs_no_improve = 0
                best_model = model
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping for rmse at epoch {epoch + 1}")
                break

        writer.close()

        fold_results_r2.append(r2)
        fold_results_rmse.append(rmse)

    # Average R^2 and RMSE across folds
    mean_r2 = np.mean(fold_results_r2)
    mean_rmse = np.mean(fold_results_rmse)

    print(f"Mean R^2 across folds: {mean_r2:.4f}")
    print(f"Mean RMSE across folds: {mean_rmse:.4f}")
    if best_model is not None:
        torch.save(best_model.state_dict(), f'outs{suffix}/best_model{suffix}.pth')

    return mean_r2, mean_rmse

def holdout_performance(holdout_dataset,batch_size,clip_model,device,suffix):
    
    # Create a DataLoader for the holdout set
    holdout_loader = DataLoader(holdout_dataset, batch_size=batch_size, shuffle=False)

    # Final model evaluation on the holdout set
    # Initialize the model
    model = CLIPRegressionModel(clip_model).to(device)
    # Load the saved state dictionary
    model.load_state_dict(torch.load(f'outs{suffix}/best_model{suffix}.pth',weights_only=True))
    # Set the model to evaluation mode
    model.eval()

    holdout_targets = []
    holdout_predictions = []

    with torch.no_grad():
        for batch in holdout_loader:
            inputs, targets = batch
            pixel_values = inputs['pixel_values'].squeeze(1).to(device)
            input_ids = inputs['input_ids'].squeeze(1).to(device)
            attention_mask = inputs['attention_mask'].squeeze(1).to(device)
            targets = targets.to(device)

            outputs = model(pixel_values, input_ids, attention_mask)

            holdout_targets.append(targets.cpu().numpy())
            holdout_predictions.append(outputs.cpu().numpy())
            #visualize_attention(clip_model, inputs, layer_num=0, is_vision=True)
            #visualize_attention(model.clip_model, inputs.squeeze(1))

    holdout_targets = np.concatenate(holdout_targets, axis=0)
    holdout_predictions = np.concatenate(holdout_predictions, axis=0)

    holdout_rmse = np.sqrt(mean_squared_error(holdout_targets, holdout_predictions))
    holdout_r2 = sklearn_r2(holdout_targets, holdout_predictions)
    print(f"Holdout R^2: {holdout_r2:.4f}")
    print(f"Holdout RMSE: {holdout_rmse:.4f}")

    return holdout_r2,holdout_rmse