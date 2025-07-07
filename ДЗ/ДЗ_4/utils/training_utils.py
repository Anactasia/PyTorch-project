import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def measure_gradient_norm(model):
    total_norm = 0.0
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
            count += 1
    return (total_norm ** 0.5) if count > 0 else 0.0

def run_epoch(model, data_loader, criterion, optimizer=None, device='cpu', is_test=False):
    if is_test:
        model.eval()
    else:
        model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    batch_grad_norms = []
    
    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)
        
        if not is_test and optimizer is not None:
            optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        
        if not is_test and optimizer is not None:
            loss.backward()
            
            # Измеряем норму градиента для этого батча
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            batch_grad_norms.append(total_norm)
            
            optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    if is_test:
        return total_loss / len(data_loader), correct / total, None
    else:
        return total_loss / len(data_loader), correct / total, batch_grad_norms


def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, train_accs = [], []
    all_grad_norms = []
    test_losses, test_accs = [], []
    
    for epoch in range(epochs):
        train_loss, train_acc, grad_norms = run_epoch(model, train_loader, criterion, optimizer, device, is_test=False)
        test_loss, test_acc, _ = run_epoch(model, test_loader, criterion, None, device, is_test=True)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        if grad_norms is not None:
            all_grad_norms.extend(grad_norms)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        print('-' * 50)
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'grad_norms': all_grad_norms  
    }


def measure_gradient_flow(model):
    total_norm = 0.0
    parameters = [p for p in model.parameters() if p.grad is not None]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5