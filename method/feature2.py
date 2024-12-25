import os
import pandas as pd
import numpy as np

def process_maf_files(input_parent_folder, output_parent_folder):
    # 遍历输入父文件夹中的子文件夹
    for foldername in os.listdir(input_parent_folder):
        input_folder = os.path.join(input_parent_folder, foldername)
        output_folder = os.path.join(output_parent_folder, foldername)

        # 如果输出文件夹不存在，则创建
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 遍历当前子文件夹中的 .maf 文件
        for filename in os.listdir(input_folder):
            if filename.endswith(".maf"):
                input_filepath = os.path.join(input_folder, filename)
                output_filename = os.path.splitext(filename)[0] + ".csv"
                output_filepath = os.path.join(output_folder, output_filename)

                # 读取 MAF 文件
                df = pd.read_csv(input_filepath, sep='\t', skiprows=7)

                # 提取所需字段
                columns_to_keep = ["1000G_AF", "1000G_AFR_AF", "1000G_AMR_AF", "1000G_EAS_AF", "1000G_EUR_AF", "1000G_SAS_AF",
                                   "ESP_AA_AF", "ESP_EA_AF", "gnomAD_AF", "gnomAD_AFR_AF", "gnomAD_AMR_AF", "gnomAD_ASJ_AF",
                                   "gnomAD_EAS_AF", "gnomAD_FIN_AF", "gnomAD_NFE_AF", "gnomAD_OTH_AF", "gnomAD_SAS_AF",
                                   "MAX_AF", "gnomAD_non_cancer_AF", "gnomAD_non_cancer_AFR_AF", "gnomAD_non_cancer_AMI_AF",
                                   "gnomAD_non_cancer_AMR_AF", "gnomAD_non_cancer_ASJ_AF", "gnomAD_non_cancer_EAS_AF",
                                   "gnomAD_non_cancer_FIN_AF", "gnomAD_non_cancer_MID_AF", "gnomAD_non_cancer_NFE_AF",
                                   "gnomAD_non_cancer_OTH_AF", "gnomAD_non_cancer_SAS_AF", "gnomAD_non_cancer_MAX_AF_adj",
                                   "Reference_Allele", "Tumor_Seq_Allele2"]  # 添加参考基因和突变基因列
                df = df[columns_to_keep]

                # 新增六列并根据参考基因和突变基因设置值
                df['C_to_A'] = np.where((df['Reference_Allele'] == 'C') & (df['Tumor_Seq_Allele2'] == 'A'), 1, 0)
                df['C_to_G'] = np.where((df['Reference_Allele'] == 'C') & (df['Tumor_Seq_Allele2'] == 'G'), 2, 0)
                df['C_to_T'] = np.where((df['Reference_Allele'] == 'C') & (df['Tumor_Seq_Allele2'] == 'T'), 3, 0)
                df['T_to_A'] = np.where((df['Reference_Allele'] == 'T') & (df['Tumor_Seq_Allele2'] == 'A'), 4, 0)
                df['T_to_C'] = np.where((df['Reference_Allele'] == 'T') & (df['Tumor_Seq_Allele2'] == 'C'), 5, 0)
                df['T_to_G'] = np.where((df['Reference_Allele'] == 'T') & (df['Tumor_Seq_Allele2'] == 'G'), 6, 0)

                # 将非数值转换为数值，然后填充为 0
                df = df.apply(pd.to_numeric, errors='coerce')
                df.fillna(0, inplace=True)
                df.replace(np.nan, 0, inplace=True)

                # 存储处理后的文件为 CSV 格式
                df.to_csv(output_filepath, index=False)

                print(f"Processed file '{filename}' saved to '{output_filepath}'.")

# 定义输入和输出父文件夹路径
input_parent_folder = "./type-maf2"
output_parent_folder = "./feature2"
# 执行处理
process_maf_files(input_parent_folder, output_parent_folder)
