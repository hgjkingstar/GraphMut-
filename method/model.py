import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# 自定义数据集类，将图像和特征数据结合在一起
class CombinedDataset(Dataset):
    def __init__(self, image_dir, feature_dir, transform=None, max_rows=150, max_cols=36):
        self.image_dataset = ImageFolder(image_dir, transform=transform)
        self.feature_dataset = self.load_features(feature_dir, max_rows, max_cols)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([label for _, label in self.feature_dataset])

    def load_features(self, data_dir, max_rows, max_cols):
        samples = []
        for i, disease_folder in enumerate(os.listdir(data_dir)):
            disease_path = os.path.join(data_dir, disease_folder)
            if os.path.isdir(disease_path):
                for sample_file in os.listdir(disease_path):
                    if sample_file.endswith('.csv'):
                        sample_path = os.path.join(disease_path, sample_file)
                        features = pd.read_csv(sample_path).values.astype(np.float32)
                        num_rows, num_cols = features.shape
                        if num_rows < max_rows:
                            padding_rows = max_rows - num_rows
                            padding = np.zeros((padding_rows, num_cols), dtype=np.float32)
                            features = np.concatenate((features, padding), axis=0)
                        elif num_rows > max_rows:
                            features = features[:max_rows, :]
                        if num_cols < max_cols:
                            padding_cols = max_cols - num_cols
                            padding = np.zeros((max_rows, padding_cols), dtype=np.float32)
                            features = np.concatenate((features, padding), axis=1)
                        elif num_cols > max_cols:
                            features = features[:, :max_cols]
                        samples.append((torch.tensor(features), i))
        return samples

    def __len__(self):
        return len(self.image_dataset)  # Assuming image and feature datasets are the same length

    def __getitem__(self, idx):
        image, _ = self.image_dataset[idx]
        features, label = self.feature_dataset[idx]
        return image, features, label
class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead=4, num_encoder_layers=3, dim_feedforward=128):
        super(TransformerModel, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_encoder_layers
        )

    def forward(self, src):
        # src shape: (seq_len, batch_size, input_dim)
        output = self.transformer(src)
        return output


class MultiModalModelWithTransformer(nn.Module):
    def __init__(self, num_classes, num_sparse_features, transformer_input_dim):
        super(MultiModalModelWithTransformer, self).__init__()
        self.image_branch = models.resnet18(pretrained=True)
        num_ftrs = self.image_branch.fc.in_features
        self.image_branch.fc = nn.Linear(num_ftrs, 128)

        self.transformer_branch = TransformerModel(transformer_input_dim)

        self.fc_final = nn.Linear(128 + 36, num_classes)

    def forward(self, image_input, transformer_input):
        image_output = self.image_branch(image_input)
        # print(f'Image output shape: {image_output.shape}')
        transformer_output = self.transformer_branch(transformer_input)
        # print(f'Transformer output shape before pooling: {transformer_output.shape}')
        # Perform pooling to reduce sequence length
        transformer_output = transformer_output.mean(dim=0)  # Shape: [batch_size, input_dim]
        # print(f'Transformer output shape after pooling: {transformer_output.shape}')
        # Flatten the tensor if needed
        transformer_output = transformer_output.view(transformer_output.size(0), -1)
        # print(f'Transformer output shape after flatten: {transformer_output.shape}')

        combined_output = torch.cat((image_output, transformer_output), dim=1)
        # print(f'Combined output shape: {combined_output.shape}')

        return self.fc_final(combined_output)
# 数据路径
image_data_dir = './type-maf/mydataset_265'
feature_data_dir = './feature2'

# 创建数据集
combined_dataset = CombinedDataset(image_data_dir, feature_data_dir, transform=transforms.ToTensor())

# 划分数据集
train_size = int(0.7 * len(combined_dataset))
test_size = len(combined_dataset) - train_size
train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 其他设置
feature_input_dim = combined_dataset.feature_dataset[0][0].shape[1]
num_classes = len(combined_dataset.label_encoder.classes_)

# 模型、损失函数、优化器
model = MultiModalModelWithTransformer(num_classes, num_sparse_features=feature_input_dim, transformer_input_dim=feature_input_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 训练和测试循环
train_loss_values = []
train_accuracy_values = []
test_loss_values = []
test_accuracy_values = []
train_f1_values = []
test_f1_values = []

for epoch in range(50):
    model.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0
    predicted_train = []
    true_train = []

    for image_inputs, feature_inputs, labels in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/50'):
        optimizer.zero_grad()
        feature_inputs = feature_inputs.permute(1, 0, 2)
        outputs = model(image_inputs, feature_inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * image_inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
        predicted_train.extend(predicted.cpu().numpy())
        true_train.extend(labels.cpu().numpy())

    epoch_train_loss = running_train_loss / total_train
    epoch_train_accuracy = correct_train / total_train
    train_loss_values.append(epoch_train_loss)
    train_accuracy_values.append(epoch_train_accuracy)
    epoch_train_f1 = f1_score(np.array(true_train), np.array(predicted_train), average='weighted')
    train_f1_values.append(epoch_train_f1)

    model.eval()
    running_test_loss = 0.0
    correct_test = 0
    total_test = 0
    predicted_test = []
    true_test = []

    with torch.no_grad():
        for image_inputs, feature_inputs, labels in tqdm(test_loader, desc='Testing'):
            feature_inputs = feature_inputs.permute(1, 0, 2)
            outputs = model(image_inputs, feature_inputs)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item() * image_inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)
            predicted_test.extend(predicted.cpu().numpy())
            true_test.extend(labels.cpu().numpy())

    epoch_test_loss = running_test_loss / total_test
    epoch_test_accuracy = correct_test / total_test
    test_loss_values.append(epoch_test_loss)
    test_accuracy_values.append(epoch_test_accuracy)
    epoch_test_f1 = f1_score(np.array(true_test), np.array(predicted_test), average='weighted')
    test_f1_values.append(epoch_test_f1)

    scheduler.step()
    print(f'Epoch [{epoch + 1}/50], Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}, Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {epoch_test_accuracy:.4f}')

# 保存模型
torch.save(model.state_dict(), './model_info/model_265_36_with_transformer_50epoch.pt')

# 绘制图表
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.plot(train_accuracy_values, label='Train Accuracy')
plt.plot(test_accuracy_values, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_loss_values, label='Train Loss')
plt.plot(test_loss_values, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(train_f1_values, label='Train F1 Score')
plt.plot(test_f1_values, label='Test F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('Training and Testing F1 Score')
plt.legend()

plt.tight_layout()
plt.show()
