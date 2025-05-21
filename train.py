import os 
from tqdm import tqdm 
import torch 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import KeyPointDataset
from model import CategoryKeypointNet
from loss import HeatmapLoss,CrossEntropyLoss
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter


# object_name = "strip"
# size = (1344,96)

# object_name = "rectangle"
# size = (480,240)

object_name = "square"
size = (320,320)


# Directory paths
image_dir = f"data/{object_name}/train_data"
label_dir = f"data/{object_name}/train_label"
save_dir = f"data/{object_name}/output"
os.makedirs(save_dir, exist_ok=True)

# Data loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.], std=[1.])
])

dataset = KeyPointDataset(image_dir, label_dir, size=size, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# TensorBoard writer
writer = SummaryWriter(log_dir='logs') 

# Model initialization
model = CategoryKeypointNet(n_channels=1, n_classes=3, n_heatmap=2, bilinear=True).cuda()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Loss function
# criterion1 = CrossEntropyLoss()
criterion2 = HeatmapLoss(loss_type="focal")

# Training parameters
num_epochs = 30
step = 0
lr_scheduler = {
    2: 0.001,
    6: 0.0005,
    7: 0.0002,
    19: 0.0001,
    24: 0.00005,
    28: 0.00001
}

# Training loop
for epoch in tqdm(range(num_epochs)):
    for image, label_heatmap, label_cls in dataloader:
        optimizer.zero_grad()
        mask = image != 0
        logits, heatmap = model(image)
        # loss_cls = criterion1(logits, label_cls)
        loss_heatmap = criterion2(heatmap, label_heatmap, mask, writer=writer)
        loss = loss_heatmap
        # loss = loss_cls + loss_heatmap
        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss/train', loss.item(), step)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], step)
        
        step += 1
        if step % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{step}], Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    if epoch in lr_scheduler:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_scheduler[epoch]  # Set to fixed learning rate

    torch.save(model.state_dict(), f'{save_dir}/model_parameters_{epoch}.pth')

writer.close()