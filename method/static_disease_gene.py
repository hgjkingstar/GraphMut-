import os
import pandas as pd

# 设置疾病文件夹路径
disease_folder = './type-maf'
# 设置保存结果的文件夹路径
result_folder = './disease_genes'

# 遍历疾病文件夹
for disease_name in os.listdir(disease_folder):
    disease_path = os.path.join(disease_folder, disease_name)
    if os.path.isdir(disease_path):
        # 初始化存储该疾病基因名的列表
        gene_names = []

        # 遍历疾病文件夹下的所有maf文件
        for filename in os.listdir(disease_path):
            if filename.endswith('.maf'):
                maf_path = os.path.join(disease_path, filename)

                # 读取maf文件，只保留变异类型为SNP、INS、DEL的基因名
                maf_data = pd.read_csv(maf_path, sep='\t', skiprows=7)
                selected_genes = maf_data.loc[maf_data['Variant_Type'].isin(['SNP', 'INS', 'DEL']), 'Hugo_Symbol'].tolist()

                # 添加到该疾病的基因名列表中
                gene_names.extend(selected_genes)

        # 去重并排序基因名列表
        gene_names = sorted(list(set(gene_names)))

        # 将结果保存到txt文件中
        result_path = os.path.join(result_folder, f'{disease_name}.txt')
        with open(result_path, 'w') as f:
            f.write('\n'.join(gene_names))
