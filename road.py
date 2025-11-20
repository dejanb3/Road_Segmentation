
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1=self.inc(x); x2=self.down1(x1); x3=self.down2(x2); x4=self.down3(x3); x5=self.down4(x4)
        x=self.up1(x5); x=torch.cat([x, x4], dim=1); x=self.conv1(x)
        x=self.up2(x); x=torch.cat([x, x3], dim=1); x=self.conv2(x)
        x=self.up3(x); x=torch.cat([x, x2], dim=1); x=self.conv3(x)
        x=self.up4(x); x=torch.cat([x, x1], dim=1); x=self.conv4(x)
        return self.outc(x)


class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and not f.startswith('.')]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        
        
        mask_name = img_name.replace('slika', 'maska', 1)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        
        mask[mask > 0] = 1.0
        
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        return image, mask


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train(); total_loss = 0
    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device); masks = masks.unsqueeze(1).to(device)
        outputs = model(images); loss = criterion(outputs, masks)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval(); total_loss = 0; dice_score = 0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images = images.to(device); masks = masks.unsqueeze(1).to(device)
            outputs = model(images); loss = criterion(outputs, masks)
            total_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            dice_score += dice_coefficient(preds, masks)
    return total_loss / len(dataloader), dice_score / len(dataloader)

def dice_coefficient(pred, target, smooth=1e-6):
    pred = pred.contiguous().view(-1); target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, save_path):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}'); print('-' * 40)
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, dice_score = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Dice Score: {dice_score:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_loss': val_loss}, save_path)
            print(f'Model saved! Best val loss: {best_val_loss:.4f}')

def predict_large_image_sliding_window(model, image_path, device, patch_size=256, stride=128):
    model.eval()
    transform = A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])
    image = np.array(Image.open(image_path).convert("RGB"))
    original_height, original_width, _ = image.shape
    full_mask = np.zeros((original_height, original_width), dtype=np.float32)
    overlap_count = np.zeros((original_height, original_width), dtype=np.float32)
    for y in range(0, original_height, stride):
        for x in range(0, original_width, stride):
            y_end = min(y + patch_size, original_height); x_end = min(x + patch_size, original_width)
            y_start = max(0, y_end - patch_size); x_start = max(0, x_end - patch_size)
            patch = image[y_start:y_end, x_start:x_end]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size: continue
            patch_tensor = transform(image=patch)['image'].unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(patch_tensor)
                pred_patch_prob = torch.sigmoid(output).cpu().numpy()[0, 0]
            full_mask[y_start:y_end, x_start:x_end] += pred_patch_prob
            overlap_count[y_start:y_end, x_start:x_end] += 1
    overlap_count[overlap_count == 0] = 1
    final_mask_prob = full_mask / overlap_count
    return (final_mask_prob > 0.5).astype(np.uint8) * 255

def visualize_prediction(image_path, mask, save_path=None):
    image = Image.open(image_path); image_np = np.array(image)
    original_height, original_width = image_np.shape[:2]
    mask_height, mask_width = mask.shape
    if original_height != mask_height or original_width != mask_width:
        mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image); axes[0].set_title('Originalna slika'); axes[0].axis('off')
    axes[1].imshow(mask, cmap='gray'); axes[1].set_title('Finalna Maska'); axes[1].axis('off')
    overlay = image_np.copy(); mask_colored = np.zeros_like(overlay)
    mask_colored[:,:,0] = mask
    overlay = np.array(image) * 0.6 + mask_colored * 0.4
    axes[2].imshow(overlay.astype(np.uint8)); axes[2].set_title('Preklopljeno'); axes[2].axis('off')
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.show()

def load_trained_model(model_path, device):
    model = UNet(n_channels=3, n_classes=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model učitan! Epoha: {checkpoint.get('epoch', 'N/A')}, Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    return model

# ====================== FUNKCIJA ZA POST-OBRADU ("RESTAURACIJA") ======================
def post_process_mask(mask):
    if mask.dtype != 'uint8' or np.max(mask) <= 1:
        mask_for_cv = (mask > 0.5).astype(np.uint8) * 255
    else:
        mask_for_cv = mask
    min_area_threshold = 7000#OVDE
    cleaned_mask_phase1 = np.zeros_like(mask_for_cv)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_for_cv, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_threshold:
            cleaned_mask_phase1[labels == i] = 255
    close_kernel_size = 11; close_iterations = 8
    close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
    final_mask = cv2.morphologyEx(cleaned_mask_phase1, cv2.MORPH_CLOSE, close_kernel, iterations=close_iterations)
    return final_mask

# ====================== FUNKCIJA ZA EVALUACIJU NA TEST SKUPU ======================
def evaluate_on_test_set(model, test_loader, device):
    print("\n--- Počinje finalna evaluacija na Test Skupu ---")
    model.eval()
    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device); masks = masks.to(device)
            outputs = model(images)
            raw_preds = torch.sigmoid(outputs) > 0.5
            cleaned_preds_list = []
            for i in range(raw_preds.shape[0]):
                raw_pred_np = raw_preds[i].cpu().numpy().squeeze().astype(np.uint8)
                cleaned_pred_np = post_process_mask(raw_pred_np)
                cleaned_preds_list.append(torch.from_numpy(cleaned_pred_np / 255.0).float())
            preds = torch.stack(cleaned_preds_list).to(device)
            preds = preds.view(-1); masks = masks.view(-1)
            total_tp += (preds * masks).sum().item()
            total_fp += ((1 - masks) * preds).sum().item()
            total_tn += ((1 - masks) * (1 - preds)).sum().item()
            total_fn += (masks * (1 - preds)).sum().item()
    epsilon = 1e-6
    dice_score = (2. * total_tp + epsilon) / (2. * total_tp + total_fp + total_fn + epsilon)
    iou_score = (total_tp + epsilon) / (total_tp + total_fp + total_fn + epsilon)
    precision = (total_tp + epsilon) / (total_tp + total_fp + epsilon)
    recall = (total_tp + epsilon) / (total_tp + total_fn + epsilon)
    return {"Dice Score": dice_score, "IoU": iou_score, "Preciznost": precision, "Odziv": recall}

def save_results_to_file(results, file_path):
    with open(file_path, 'w') as f:
        f.write("Finalni rezultati evaluacije modela (na očišćenim maskama):\n" + "="*60 + "\n")
        for key, value in results.items(): f.write(f"{key}: {value:.4f}\n")
        f.write("="*60 + "\n")
    print(f"Rezultati sačuvani u fajl: {file_path}")



# ============== FUNKCIJA ZA TRENIRANJE ==============
def run_training():
    """Ova funkcija radi SAMO treniranje."""
    print("\n--- Počinje proces treniranja ---")
    BATCH_SIZE = 16; NUM_EPOCHS = 50; LEARNING_RATE = 1e-4
    IMAGE_HEIGHT = 256; IMAGE_WIDTH = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Koristi se: {device}')
    DRIVE_FOLDER = "/content/drive/MyDrive/"
    TRAIN_IMG_DIR = os.path.join(DRIVE_FOLDER, "data/train/images"); TRAIN_MASK_DIR = os.path.join(DRIVE_FOLDER, "data/train/maska")
    VAL_IMG_DIR = os.path.join(DRIVE_FOLDER, "data/val/images"); VAL_MASK_DIR = os.path.join(DRIVE_FOLDER, "data/val/maska")
    train_transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.2), A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.5), A.GaussNoise(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2(),
    ])
    train_dataset = RoadDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
    val_dataset = RoadDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    model = UNet(n_channels=3, n_classes=1).to(device)
    save_model_path = os.path.join(DRIVE_FOLDER, 'unet_road_segmentation.pth')
    train_model(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, device, save_model_path)
    print("\n--- Trening završen ---")

# ============== FUNKCIJA ZA FINALNO TESTIRANJE ==============
def run_final_evaluation():
    """Ova funkcija radi SAMO evaluaciju na test skupu."""
    print("\n--- Počinje proces finalne evaluacije ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DRIVE_FOLDER = "/content/drive/MyDrive/"
    TEST_IMG_DIR = os.path.join(DRIVE_FOLDER, "data/test/images")
    TEST_MASK_DIR = os.path.join(DRIVE_FOLDER, "data/test/maska")
    save_model_path = os.path.join(DRIVE_FOLDER, 'unet_road_segmentation.pth')
    test_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2(),
    ])
    test_dataset = RoadDataset(TEST_IMG_DIR, TEST_MASK_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
    best_model = load_trained_model(save_model_path, device)
    test_results = evaluate_on_test_set(best_model, test_loader, device)
    print("\n" + "="*40 + "\n      FINALNI REZULTATI NA TEST SKUPU\n" + "="*40)
    for key, value in test_results.items(): print(f"{key}: {value:.4f}")
    print("="*40)
    results_file_path = os.path.join(DRIVE_FOLDER, 'final_evaluation_results.txt')
    save_results_to_file(test_results, results_file_path)
    print("\n--- Evaluacija završena ---")

# ============== FUNKCIJA ZA PREDIKCIJU NA JEDNOJ SLICI ==============
def example_usage():
    """Ova funkcija radi SAMO predikciju na jednoj velikoj slici."""
    print("\n--- Počinje proces predikcije na jednoj slici ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DRIVE_FOLDER = '/content/drive/MyDrive/'
    model_path = os.path.join(DRIVE_FOLDER, 'unet_road_segmentation.pth') 
    model = load_trained_model(model_path, device)
    test_image_path = os.path.join(DRIVE_FOLDER, 'test_road.tif') 
    raw_predicted_mask = predict_large_image_sliding_window(model, test_image_path, device)
    cleaned_mask = post_process_mask(raw_predicted_mask)
    save_path_viz = os.path.join(DRIVE_FOLDER, 'prediction_RESTAURACIJA.png')
    visualize_prediction(test_image_path, cleaned_mask, save_path=save_path_viz)
    save_path_mask = os.path.join(DRIVE_FOLDER, 'predicted_mask_RESTAURACIJA.png')
    Image.fromarray(cleaned_mask).save(save_path_mask)
    print(f"Vizualizacija i finalna maska su sačuvane.")
    print("\n--- Predikcija na jednoj slici završena ---")


if __name__ == "__main__":
    
    # KORAK 1: Ako želite da trenirate model, odkomentarišite i pokrenite ovu liniju.
    # run_training()
    
    # KORAK 2: Nakon treninga, da biste dobili finalne metrike, odkomentarišite i pokrenite ovu.
    run_final_evaluation()
    
    # KORAK 3: Da biste videli rezultat na jednoj velikoj slici, odkomentarišite i pokrenite ovu.
    example_usage()