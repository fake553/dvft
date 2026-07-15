import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from timm.utils import ModelEma
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from transformers import get_cosine_schedule_with_warmup


def train_classification(model, train_loader, val_loader, device,
                         epochs=20, num_classes=1000, save_path='best_model.pth'):
    """统一的分类训练循环"""
    model = model.to(device)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    
    # MixUp
    mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=0.5,
                     label_smoothing=0.1, num_classes=num_classes)
    
    # EMA
    ema = ModelEma(model, decay=0.9998, device=device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-5, weight_decay=1e-2
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=500,
        num_training_steps=len(train_loader) * epochs
    )
    
    criterion = SoftTargetCrossEntropy()
    val_criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            images, labels = mixup_fn(images, labels)
            
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            ema.update(model)
            
            total_loss += loss.item()
        
        # Validation
        acc = evaluate(model, val_loader, device)
        acc_ema = evaluate(ema.ema, val_loader, device)
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, "
              f"Acc={acc:.2f}%, EMA Acc={acc_ema:.2f}%")
        
        if acc_ema > best_acc:
            best_acc = acc_ema
            torch.save({
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.ema.state_dict(),
                'best_acc': best_acc
            }, save_path)
    
    print(f"Best EMA accuracy: {best_acc:.2f}%")
    return best_acc


def evaluate(model, val_loader, device):
    """评估"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, pred = torch.max(logits, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    return 100 * correct / total