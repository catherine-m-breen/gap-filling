import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import os
from sklearn.model_selection import train_test_split
import time
from dictionaries import split_basin_dict, flight_to_basin, snowclass_dict, land_class_dict
import Path
import zarr

print('start')

def random_crop(data, crop_size):
    width, height = data.shape[2], data.shape[3]
    
    pad_width = max(crop_size[0] - width, 0)
    pad_height = max(crop_size[1] - height, 0)

    if pad_width > 0 or pad_height > 0:
        data = np.pad(data, ((0, 0), (0, 0), (0, pad_width), (0, pad_height)), mode='constant')

    width, height = data.shape[2], data.shape[3]
    dw = np.random.randint(0, width - crop_size[0] + 1)
    dh = np.random.randint(0, height - crop_size[1] + 1)
    
    return data[:, :, dw:dw+crop_size[0], dh:dh+crop_size[1]]

class ASODataset(Dataset):
    def __init__(self, data, labels, crop_size=(5, 5), augment=False):
        assert len(data) == len(labels), "Mismatch in lengths of data and labels"
        self.data = data
        self.labels = labels
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        label_item = self.labels[idx]
        if self.augment:
            ##### seems like both data and label should be cropped the same? ##
            data_item = random_crop(data_item, self.crop_size)
            label_item = random_crop(label_item, self.crop_size)
        return data_item, label_item

class Model(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = self._make_layer(input_channels, 16)
        self.conv2 = self._make_layer(16, 32)
        self.conv3 = self._make_layer(32, 64)
        self.conv4 = self._make_layer(64, 128)
        self.conv5 = self._make_layer(128, 64)
        self.conv6 = self._make_layer(64, 32)
        self.conv7 = self._make_layer(32, 16)
        self.conv8 = self._make_layer(16, 8)
        self.conv100 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_uniform_(self.conv100.weight, nonlinearity='relu')

    def _make_layer(self, in_channels, out_channels, dropout_prob=0.2):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        nn.init.kaiming_uniform_(layer[0].weight, nonlinearity='relu')
        return layer

    def forward(self, x):
        # Input: (batch, channels, H, W)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv100(x)
        
        # Output: (batch, 1, H, W)
        x = x.squeeze(1)  # Remove channel dimension → (batch, H, W)
        
        return x
    

def train_model(model, dataloader, optimizer, criterion, device, epoch, batch_size=8):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for i, (features, labels) in enumerate(dataloader):
        features = features.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)

        if features.shape[3] == 1 or features.shape[4] == 1:
            continue

        output = model(features)
        mask = (labels != 0)
        labels = labels[mask]
        output = output[mask]

        l1_lambda = 0.000001
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = criterion(output, labels) + l1_lambda * l1_norm

        loss.backward()
        total_loss += loss.item()

        all_preds.extend(output.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (i + 1) % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()

    if (i + 1) % batch_size != 0:
        optimizer.step()
        optimizer.zero_grad()

    print(total_loss)
    return total_loss / len(dataloader), (all_labels, all_preds)

def validate_model(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (features, labels) in enumerate(dataloader):
            features = features.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            if features.shape[3] == 1 or features.shape[4] == 1:
                continue

            output = model(features)
            mask = (labels != 0)
            labels = labels[mask]
            output = output[mask]

            loss = criterion(output, labels)
            total_loss += loss.item()

            all_preds.extend(output.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print('number of val pixels', len(all_labels))
    return total_loss / len(dataloader), (all_labels, all_preds)

def test_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (features, labels) in enumerate(dataloader):
            features = features.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            if features.shape[3] == 1 or features.shape[4] == 1:
                continue

            output = model(features)
            mask = (labels != 0)
            labels = labels[mask]
            output = output[mask]

            all_preds.extend(output.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print('number of test pixels', len(all_labels))
    return all_labels, all_preds

def normalize_dataset_per_channel(data_list):
    normalized_data_list = []
    for data in data_list:
        data_tensor = torch.tensor(data, dtype=torch.float32)
        mean = torch.mean(data_tensor, dim=(0, 2, 3), keepdim=True)
        std = torch.std(data_tensor, dim=(0, 2, 3), keepdim=True)
        normalized_tensor = (data_tensor - mean) / (std + 1e-7)
        normalized_data = normalized_tensor.numpy()
        normalized_data_list.append(normalized_data)
    return normalized_data_list

# def load_dataset(folder_path):
#     data_list = []
#     label_list = []

#     files = sorted(os.listdir(folder_path))
#     for file in files:
#         if file.endswith("_data.npy"):
#             data_path = os.path.join(folder_path, file)
#             data = np.load(data_path)
#             data = np.concatenate((data[:, 0:1, :, :], data[:, 6:9, :, :]), axis=1)
#             data_list.append(data)

#             label_file = file.replace("_data.npy", "_yield.npy")
#             label_path = os.path.join(folder_path, label_file)
#             if os.path.exists(label_path):
#                 label = np.load(label_path)
#                 crop_type = file.split('_')[1]

#                 if crop_type == "corn":
#                     label *= 12.454
#                 elif crop_type in ["soy", "wheat"]:
#                     label *= 13.34368

#                 if label.shape[0] > 2 and label.shape[1] > 2:
#                     label[0:1, :] = 0
#                     label[:, 0:1] = 0
#                     label[:, -1] = 0
#                     label[-1, :] = 0
#                 elif label.shape[0] <= 2 and label.shape[1] > 2:
#                     label[:, 0:1] = 0
#                     label[:, -1] = 0
#                 elif label.shape[0] > 2 and label.shape[1] <= 2:
#                     label[0:1, :] = 0
#                     label[-1, :] = 0

#                 label_list.append(label)

#     return data_list, label_list

def load_dataset_test(folder_path):
    data_list = []
    files = sorted(os.listdir(folder_path))
    for file in files:
        if file.endswith("_data.npy"):
            data_path = os.path.join(folder_path, file)
            data = np.load(data_path)
            data = np.concatenate((data[:, 0:1, :, :], data[:, 6:9, :, :]), axis=1)
            data_list.append(data)
            print(len(data_list))
    return data_list

def compute_nrmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) / (np.max(y_true) - np.min(y_true))

def compute_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def compute_me(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

def compute_d(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true))) ** 2))

def add_gaussian_noise(image, mean=0, sigma=25):
    time, ch, row, col = image.shape
    noisy = np.zeros_like(image)
    for t in range(time):
        gauss = np.random.normal(mean, sigma, (ch, row, col))
        noisy[t] = image[t] + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    zarr_dir = "/discover/nobackup/cmbreen/gap-filling-data/zarr_chunks"
    zarr_files = list(Path(zarr_dir).glob("*.zarr"))

    ## this will eventually be load dataset ##
    basins = []
    tif_names = []
    splits = []
    basenames = []
    x_data = []
    y_data = []
    patches = []
    ## going through all the zarr patches and get the information
    for zarr_path in sorted(zarr_files):
        tif_name = zarr_path.stem + '.tif'
        basin = flight_to_basin[tif_name]
        split = split_basin_dict[basin]
        z = zarr.open(str(zarr_path), mode='r')
        X = z['X']
        ### mask unwanted pixels
        ## leave out for now
        Y = z['Y']

        _, height, width = X.shape
        patches = []
        patch_size = 256 
        stride = 128
        for row in range(0, height - patch_size + 1, stride):
            for col in range(0, width - patch_size + 1, stride):
                patches.append((row, col))
                tif_names.append(tif_name)
                basins.append(basins), splits.append(split), basenames.append(zarr_path.stem)
                x_data.append(X), y_data.append(Y)


    data = {'basins':basin, 'split':split, 'basenames':basenames,
            'x_data':x_data}
    labels = y_data

    ##################

    #data, labels = load_dataset(zarr_files)
    ## normalize train and val together
    data = normalize_dataset_per_channel(data)

    ## normalize test 
    test_data = load_dataset_test('test_dataset')
    test_data = normalize_dataset_per_channel(test_data)

    ## separate into patches ## 
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.1, shuffle=True, random_state=42)

    def random_flip(data, labels):
        flip_w = np.random.choice([True, False])
        flip_h = np.random.choice([True, False])
        
        if flip_w:
            data = np.flip(data, axis=2).copy()
            labels = np.flip(labels, axis=0).copy()
        if flip_h:
            data = np.flip(data, axis=3).copy()
            labels = np.flip(labels, axis=1).copy()
        
        return
    augmented_data = []
    augmented_labels = []

    for data, labels in zip(train_data, train_labels):
        flipped_data, flipped_labels = random_flip(data, labels)
        noisy_data_gaussian = add_gaussian_noise(flipped_data)
        augmented_data.append(noisy_data_gaussian)
        augmented_labels.append(flipped_labels)

    augmented_data = np.array(augmented_data)
    augmented_labels = np.array(augmented_labels)

    combined_train_data = np.concatenate((train_data, augmented_data), axis=0)
    combined_train_labels = np.concatenate((train_labels, augmented_labels), axis=0)

    combined_dataset = ASODataset(combined_train_data, combined_train_labels, augment=False)
    val_dataset = ASODataset(val_data, val_labels, augment=False)

    print(f"Train dataset size: {len(combined_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_dataloader = DataLoader(combined_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = ASODataset(input_channels=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.000001)
    criterion = nn.L1Loss()
    scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=0.0001)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 400
    epochs_without_improvement = 0

    for epoch in range(1000):
        train_loss, _ = train_model(model, train_dataloader, optimizer, criterion, device, epoch)
        val_loss, _ = validate_model(model, val_dataloader, criterion, device, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_weights_CNN_test.pth')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping!")
            break

    model.load_state_dict(torch.load('best_model_weights_CNN_corn.pth', map_location=device))

    columns = ['Corn', 'Soy', 'Wheat']
    years = ['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']

    fig, axs = plt.subplots(len(years), len(columns), figsize=(15, 40), dpi=300)
    wheat_pred = []
    wheat_value = []
    soy_pred = []
    soy_value = []
    corn_pred = []
    corn_value = []

    corn_value = np.array([217, 180, 305])
    soy_value = np.array([78, 70, 64, 71])

    corn_value = corn_value * 12.454 * 5.039911111
    soy_value = soy_value * 13.34368 * 5.039911111

    labels = np.concatenate((corn_value, soy_value))

    for i, features in enumerate(test_data):
        idx = i
        features = torch.tensor(features).unsqueeze(0).to(device, dtype=torch.float32)
        labels_tensor = torch.tensor(labels).unsqueeze(0).to(device, dtype=torch.float32)

        output = model(features)
        mask = (features != 0)

        a = mask[0, 0, 0, :, :].unsqueeze(0)
        output2 = output[a].unsqueeze(0)

        if idx < 3:
            corn_pred.append(torch.mean(output2).item() * 5.039911111)
        else:
            soy_pred.append(torch.mean(output2).item() * 5.039911111)

    soy_pred = np.array(soy_pred)
    soy_value = np.array(soy_value)
    corn_pred = np.array(corn_pred)
    corn_value = np.array(corn_value)

    mae = mean_absolute_error(soy_value, soy_pred)
    rmse = np.sqrt(mean_squared_error(soy_value, soy_pred))
    r2 = r2_score(soy_value, soy_pred)
    NRMSE = np.sqrt(mean_squared_error(soy_value, soy_pred)) / (np.max(soy_value) - np.min(soy_value))
    ME = 1 - (np.sum((soy_value - soy_pred) ** 2) / np.sum((soy_value - np.mean(soy_value)) ** 2))
    d = 1 - (np.sum((soy_value - soy_pred) ** 2) / np.sum((np.abs(soy_pred - np.mean(soy_value)) + np.abs(soy_value - np.mean(soy_value))) ** 2))
    mape = np.mean(np.abs((soy_value - soy_pred) / soy_value)) * 100

    print('soy', mae, rmse, r2, NRMSE, ME, d, mape)

    mae = mean_absolute_error(corn_value, corn_pred)
    rmse = np.sqrt(mean_squared_error(corn_value, corn_pred))
    r2 = r2_score(corn_value, corn_pred)
    NRMSE = np.sqrt(mean_squared_error(corn_value, corn_pred)) / (np.max(corn_value) - np.min(corn_value))
    ME = 1 - (np.sum((corn_value - corn_pred) ** 2) / np.sum((corn_value - np.mean(corn_value)) ** 2))
    d = 1 - (np.sum((corn_value - corn_pred) ** 2) / np.sum((np.abs(corn_pred - np.mean(corn_value)) + np.abs(corn_value - np.mean(corn_value))) ** 2))
    mape = np.mean(np.abs((corn_value - corn_pred) / corn_value)) * 100

    print('corn', mae, rmse, r2, NRMSE, ME, d, mape)

if __name__ == '__main__':
    main()