import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, confusion_matrix, accuracy_score,
    precision_score, recall_score, classification_report, log_loss
)

# device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

def run_epoch(loader, model, criterion, optimizer=None, train=True, device='cpu'):
    """Run one epoch of training or evaluation"""
    # set the model to training or eval mode
    if train:
        model.train()
    else:
        model.eval()
    
    losses = []
    all_preds = []
    all_targets = []
    
    for images, targets in loader:
        # TODO: Move to device
        images = images.to(device)
        targets = targets.to(device)
        
        with torch.set_grad_enabled(train):
            # TODO: Forward pass
            logits = model(images)
            loss = criterion(logits, targets)
        
        if train:
            # TODO: Backward pass
            # do a backward pass to get gradients
            # do i do optimizer.zero_grad()?
            loss.backward()
            # update weights with Adam optimizer
            optimizer.step()
            
        
        # TODO: Store predictions and targets
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        # add probs for class 1 to all_preds list
        all_preds.extend(probs.tolist())
        # add targets to alltargets list
        all_targets.extend(targets.detach().cpu().numpy().tolist())
        losses.append(loss.item())
    
    # TODO: Compute metrics
    y_true = np.array(all_targets)
    y_prob = np.array(all_preds)
    y_pred = (y_prob >= 0.5).astype(int)
    
    # TODO: Compute accuracy, sensitivity, specificity, and AUC
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn/(tn + fp)
    
    try:
        auc_val = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_val = np.nan
    
    return np.mean(losses), acc, sens, spec, auc_val