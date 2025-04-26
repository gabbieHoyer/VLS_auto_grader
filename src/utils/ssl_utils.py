import torch
from tqdm import tqdm

def ssl_pretrain(model, dataloader, criterion, optimizer, epochs, device, logger):
    """
    Pre-train the model using self-supervised learning (contrastive loss).
    
    Args:
        model: The model to pre-train (I3D or ViViT in SSL mode).
        dataloader: DataLoader with two augmented views per video.
        criterion: Contrastive loss function.
        optimizer: Optimizer for pre-training.
        epochs: Number of pre-training epochs.
        device: Device to run on (cuda or cpu).
        logger: Logger for logging progress.
    """
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for view1, view2 in tqdm(dataloader, desc=f"SSL Pre-training Epoch {epoch+1}/{epochs}"):
            view1, view2 = view1.to(device), view2.to(device)
            
            optimizer.zero_grad()
            z1 = model(view1)  # Embeddings for view 1
            z2 = model(view2)  # Embeddings for view 2
            
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        logger.info(f"SSL Pre-training Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")