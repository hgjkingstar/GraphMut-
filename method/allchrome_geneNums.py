import os


def count_genes_in_maf(maf_path):
    genes = {}
    with open(maf_path, 'r') as maf_file:
        # Skip the first 7 lines (annotation)
        for _ in range(7):
            next(maf_file)
        header = next(maf_file).strip().split('\t')
        hugo_symbol_index = header.index('Hugo_Symbol')
        variant_type_index = header.index('Variant_Type')
        chromosome_index = header.index('Chromosome')
        for line in maf_file:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                hugo_symbol = parts[hugo_symbol_index]
                variant_type = parts[variant_type_index]
                chromosome = parts[chromosome_index]
                if variant_type in ['SNP', 'INS', 'DEL']:
                    if chromosome not in genes:
                        genes[chromosome] = set()
                    genes[chromosome].add(hugo_symbol)
    return genes


def process_disease_folder(disease_folder, all_genes):
    disease_name = os.path.basename(disease_folder)
    for sample_file in os.listdir(disease_folder):
        if sample_file.endswith('.maf'):
            sample_path = os.path.join(disease_folder, sample_file)
            sample_genes = count_genes_in_maf(sample_path)
            for chromosome, genes in sample_genes.items():
                if chromosome not in all_genes:
                    all_genes[chromosome] = set()
                all_genes[chromosome].update(genes)


def write_genes_to_files(all_genes, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for chromosome, genes in all_genes.items():
        output_file = os.path.join(output_folder, f'{chromosome}.txt')
        with open(output_file, 'w') as f:
            f.write('\n'.join(sorted(genes)))
        print(f"Chromosome {chromosome}: {len(genes)} genes")


def main():
    input_folder = './type-maf'
    all_genes = {}
    for disease_folder in os.listdir(input_folder):
        disease_path = os.path.join(input_folder, disease_folder)
        if os.path.isdir(disease_path):
            process_disease_folder(disease_path, all_genes)

    output_folder = './all_diseases_genes'
    write_genes_to_files(all_genes, output_folder)


if __name__ == "__main__":
    main()
