import os
from collections import Counter

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, random_split
from torchvision import models, transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torchvision.datasets import ImageFolder
matplotlib.rcParams['font.family'] = 'Arial'



# 定义特征矩阵数据集类
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
# 设置随机种子
torch.manual_seed(32)
# 划分数据集
train_size = int(0.7 * len(combined_dataset))
test_size = len(combined_dataset) - train_size
train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])


# 读取基因名称函数
def read_gene_names(folder_path):
    chromosome_directories = ['chr1.txt', 'chr2.txt', 'chr3.txt', 'chr4.txt', 'chr5.txt', 'chr6.txt', 'chr7.txt',
                              'chr8.txt', 'chr9.txt', 'chr10.txt', 'chr11.txt', 'chr12.txt', 'chr13.txt', 'chr14.txt',
                              'chr15.txt', 'chr16.txt', 'chr17.txt', 'chr18.txt', 'chr19.txt', 'chr20.txt', 'chr21.txt',
                              'chr22.txt', 'chrX.txt', 'chrY.txt']

    gene_chrom_dict = {}

    # Open each chrom file for ACC, Chromosome 0-23
    for chrom in range(len(chromosome_directories)):
        gene_name_list = list()
        file_path = os.path.join(folder_path, chromosome_directories[chrom])
        with open(file_path) as mf:
            for line in mf:
                line = line.strip()
                cols = line.split()
                gene_name_list.append(cols[0])
        gene_chrom_dict[chrom] = gene_name_list
    gene_index_dict = {0: 'gene_index_chrom01', 1: 'gene_index_chrom02', 2: 'gene_index_chrom03',
                       3: 'gene_index_chrom04', 4: 'gene_index_chrom05', 5: 'gene_index_chrom06',
                       6: 'gene_index_chrom07',
                       7: 'gene_index_chrom08', 8: 'gene_index_chrom09', 9: 'gene_index_chrom10',
                       10: 'gene_index_chrom11',
                       11: 'gene_index_chrom12', 12: 'gene_index_chrom13', 13: 'gene_index_chrom14',
                       14: 'gene_index_chrom15',
                       15: 'gene_index_chrom16', 16: 'gene_index_chrom17', 17: 'gene_index_chrom18',
                       18: 'gene_index_chrom19',
                       19: 'gene_index_chrom20', 20: 'gene_index_chrom21', 21: 'gene_index_chrom22',
                       22: 'gene_index_chromX', 23: 'gene_index_chromY'}
    for chrom in range(len(chromosome_directories)):
        # print("chrom is ",chrom) # 0-23
        gene_dict = dict()
        gene_index = 0
        for gene in range(len(gene_chrom_dict[chrom])):
            gene_dict[gene_chrom_dict[chrom][gene]] = gene_index
            gene_index += 1
        gene_index_dict[chrom] = gene_dict

    return gene_chrom_dict, gene_index_dict

# Grad-CAM 实现
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册钩子以获取特征图和梯度
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, image_input, feature_input, target_class):
        self.model.eval()
        feature_input= feature_input.permute(1, 0, 2)
        # 获取模型的综合输出
        output = self.model(image_input, feature_input)

        # 计算目标类的损失
        loss = output[:, target_class].sum()

        # 反向传播以获取梯度
        self.model.zero_grad()
        loss.backward()

        # 获取梯度和特征图
        gradients = self.gradients.data.numpy()[0]
        activations = self.activations.data.numpy()[0]

        # 计算权重并生成Grad-CAM
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # 归一化并调整大小
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (image_input.shape[2], image_input.shape[3]))
        cam -= np.min(cam)
        cam /= np.max(cam)

        return cam


# 获取基因名称
def get_gene_name_and_values_from_coordinates(coordinates, cam, gene_chrom_dict, N):
    result = []

    # 预计算每条染色体的列偏移量
    chrom_column_offsets = []
    offset = 0

    for chrom in range(24):
        num_genes = len(gene_chrom_dict[chrom])
        ki = (num_genes + N - 1) // N  # 简化后的公式，向上取整
        chrom_column_offsets.append(offset)
        offset += ki * 3

    # 添加一个额外的偏移量来避免越界
    chrom_column_offsets.append(offset)
    print(chrom_column_offsets)

    for coord in coordinates:
        x, y = coord

        try:
            # 找到对应的染色体
            chrom = None
            for i in range(24):
                if x < chrom_column_offsets[i + 1]:
                    chrom = i
                    break

            if chrom is None:
                print(f"X coordinate {x} exceeds the total width of the mutation map")
                continue

            # 在染色体内计算基因索引
            num_genes = len(gene_chrom_dict[chrom])
            ki = (num_genes + N - 1) // N  # 使用简化后的公式

            row_in_chrom = y
            col_in_chrom = (x - chrom_column_offsets[chrom]) // 3
            gene_index = col_in_chrom + row_in_chrom * ki

            if gene_index >= num_genes:
                print(f"Calculated gene index {gene_index} exceeds the number of genes on chromosome {chrom}")
                continue

            # 获取基因名和热力图值
            gene_name = gene_chrom_dict[chrom][gene_index]
            gene_value = cam[y, x]
            result.append((gene_name, gene_value))

        except Exception as e:
            print(f"Error processing coordinate ({x, y}): {e}")
            continue

    # 按热力图值排序并去重
    result = sorted(result, key=lambda x: x[1], reverse=True)
    unique_genes = {}
    for gene_name, gene_value in result:
        if gene_name not in unique_genes:
            unique_genes[gene_name] = gene_value
    result = list(unique_genes.items())

    return result




# 特征矩阵的列数作为特征矩阵输入维度
feature_input_dim = combined_dataset.feature_dataset[0][0].shape[1]
print(feature_input_dim)
# 加载训练好的模型
num_classes = 33  # 根据您的数据集更新类别数量
model = MultiModalModelWithTransformer(num_classes, num_sparse_features=feature_input_dim,transformer_input_dim=feature_input_dim)
model.load_state_dict(torch.load('./model_info/model_265_36_with_transformer_50epoch.pt'))
model.eval()

# 选择要可视化的层
target_layer = model.image_branch.layer4[-1]

# 创建Grad-CAM对象
grad_cam = GradCAM(model, target_layer)


# 读取所有基因名称的 TXT 文件
def read_all_genes_from_txt(file_path):
    with open(file_path, 'r') as file:
        genes = [line.strip() for line in file.readlines()]
    return set(genes)  # 使用集合存储基因名称，以便快速检索


# 读取基因名称
gene_set = read_all_genes_from_txt("./disease_genes/KIRC.txt")
# 初始化一个字典来存储所有样本的驱动基因
all_samples_genes = {}
# 初始化一个列表来存储每个样本的驱动基因列表
all_samples_genes_list = []

# 初始化一个列表来存储每个样本的筛选后的基因列表
all_samples_filtered_genes = []
# 初始化一个字典来存储每个基因的热力图值
gene_heatmap_values = {}
for i in range(3567, 3979):
    # 从数据集中获取图像和特征矩阵
    image_input, feature_input, label = combined_dataset[i]
    # 选择要可视化的目标类别
    target_class = label  # 设置目标类，您可以更改为特定的类别
    cam = grad_cam(image_input.unsqueeze(0), feature_input.unsqueeze(0), target_class)
    # 应用阈值过滤
    threshold = 0.5  # 您可以根据需要调整阈值
    # cam[cam < threshold] = 0

    # 将热力图叠加到原始图像上进行显示
    original_image = Image.fromarray(np.uint8(image_input.permute(1, 2, 0).numpy() * 255))  # 将张量转换为 PIL 图像格式
    original_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_BGR2RGB)
    # 转换为RGB格式
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_output = heatmap + np.float32(original_image) / 255
    cam_output = cam_output / np.max(cam_output)
    # 显示热力图
    plt.imshow(cam_output)
    # 读取基因名称
    gene_chrom_dict, gene_index_dict = read_gene_names("./type-maf/gene_counts/all_diseases_genes")
    # 获取大于阈值的像素位置
    rows, cols = np.where(cam >= threshold)
    coordinates = [(col, row) for row, col in zip(rows, cols)]
    # 获取相关基因名称和热力图值
    result = get_gene_name_and_values_from_coordinates(coordinates, cam, gene_chrom_dict, 265)

    # 筛选出在TXT文件中存在的驱动基因
    filtered_genes = [(gene, value) for gene, value in result if gene in gene_set]
    # 更新字典中每个基因的热力图值列表
    for gene, value in filtered_genes:
        if gene in gene_heatmap_values:
            gene_heatmap_values[gene].append(value)
        else:
            gene_heatmap_values[gene] = [value]


# 收集所有基因的平均热力图值到一个列表中
heatmap_means = {}

for gene, heatmap_values in gene_heatmap_values.items():
    mean_heatmap_value = np.mean(heatmap_values)
    heatmap_means[gene]=mean_heatmap_value

print(heatmap_means)
# 将字典转换为DataFrame
df_heatmap_means = pd.DataFrame(list(heatmap_means.items()), columns=['Gene', 'MeanHeatmapValue'])

# 按照热力图值从大到小排序
df_heatmap_means = df_heatmap_means.sort_values(by='MeanHeatmapValue', ascending=False).reset_index(drop=True)



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# # 绘制热力图值的下降趋势图
# plt.figure(figsize=(10, 6))
# sns.lineplot(x=df_heatmap_means.index, y=df_heatmap_means['MeanHeatmapValue'])
# plt.xlabel('Gene Index')
# plt.ylabel('Mean Heatmap Value')
# plt.title('Descending Trend of Mean Heatmap Values')
# plt.show()
#
# # 计算相邻基因热力图值的变化量
# df_heatmap_means['ValueChange'] = df_heatmap_means['MeanHeatmapValue'].diff().abs()
#
# # 找出变化最大的基因
# max_change_index = df_heatmap_means['ValueChange'].idxmax()
# max_change_gene = df_heatmap_means.iloc[max_change_index]
#
# # 输出变化最快的基因及其位置
# print(f"Gene with the fastest decrease in value: {max_change_gene['Gene']}, Position: {max_change_index}")

# print(df_heatmap_means)
# print(df_heatmap_means[df_heatmap_means["Gene"]=="BRCA1"])
# print(df_heatmap_means[df_heatmap_means["Gene"]=="BRCA2"])
# print(df_heatmap_means[df_heatmap_means["Gene"]=="TP53"])
# print(df_heatmap_means[df_heatmap_means["Gene"]=="PIK3CA"])
# print(df_heatmap_means[df_heatmap_means["Gene"]=="PTEN"])
# print(df_heatmap_means[df_heatmap_means["Gene"]=="HER2"])
# print(df_heatmap_means[df_heatmap_means["Gene"]=="CDH1"])
# print(df_heatmap_means[df_heatmap_means["Gene"]=="GATA3"])
# print(df_heatmap_means[df_heatmap_means["Gene"]=="MAP3K1"])
# print(df_heatmap_means[df_heatmap_means["Gene"]=="AKT1"])
# print(df_heatmap_means[df_heatmap_means["Gene"]=="CCND1"])
# 绘制概率密度图
plt.figure(figsize=(3.5, 2.5),dpi=330)
plt.hist(df_heatmap_means['MeanHeatmapValue'], bins=30, density=True, color='#d0d570', edgecolor='black')
# plt.title('Density Plot of Mean Heatmap Values For KIRC',fontsize=10,fontweight='bold')
plt.xlabel('Mean Heatmap Value',fontsize=8,fontweight='bold')
plt.ylabel('Density',fontsize=8,fontweight='bold')
plt.xticks( fontsize=8,fontweight='bold')
plt.yticks(fontsize=8,fontweight='bold')
# plt.grid(True)
# plt.show()
plt.tight_layout()
plt.savefig("./image4/KIRC密度绘制.pdf")
