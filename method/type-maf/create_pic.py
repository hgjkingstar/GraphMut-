
# here put the import lib
import numpy as np 
import pandas as pd
import sys
import os
import glob
import datetime
import cv2
import colorsys

def increase_brightness(rgb_color, increase_factor):
	# 将 RGB 颜色转换成 HSV 颜色空间
	hsv_color = colorsys.rgb_to_hsv(rgb_color[0] / 255.0, rgb_color[1] / 255.0, rgb_color[2] / 255.0)

	# 增加亮度
	increased_brightness = min(hsv_color[2] * increase_factor, 1.0)  # 亮度不超过 1.0

	# 将 HSV 颜色转换回 RGB 颜色空间
	new_rgb_color = colorsys.hsv_to_rgb(hsv_color[0], hsv_color[1], increased_brightness)

	# 将浮点数的颜色值转换成整数
	new_rgb_color_int = [int(component * 255) for component in new_rgb_color]

	return new_rgb_color_int
def read_gene_names(path, cancer_sample_directory_path, cancer_name):
	os.chdir(path)
	genes = list()
	# read "chorm01.txt-chormY.txt" files
	# chromosome_directories = [name for name in os.listdir(".") if os.path.isfile(name)]
	# chromosome_directories.sort()
	chromosome_directories = ['chr1.txt', 'chr2.txt', 'chr3.txt', 'chr4.txt', 'chr5.txt', 'chr6.txt', 'chr7.txt',
							  'chr8.txt', 'chr9.txt', 'chr10.txt', 'chr11.txt', 'chr12.txt', 'chr13.txt', 'chr14.txt',
							  'chr15.txt', 'chr16.txt', 'chr17.txt', 'chr18.txt', 'chr19.txt', 'chr20.txt', 'chr21.txt',
							  'chr22.txt', 'chrX.txt', 'chrY.txt']
	print("chromosome_directories=",chromosome_directories)

	gene_chrom_dict = {0: 'gene_chrom01_list',1: 'gene_chrom02_list',2: 'gene_chrom03_list',
	3: 'gene_chrom04_list',4: 'gene_chrom05_list',5: 'gene_chrom06_list',6: 'gene_chrom07_list',
	7: 'gene_chrom08_list',8: 'gene_chrom09_list',9: 'gene_chrom10_list',10: 'gene_chrom11_list',
	11: 'gene_chrom12_list',12: 'gene_chrom13_list',13: 'gene_chrom14_list',14: 'gene_chrom15_list',
	15: 'gene_chrom16_list',16: 'gene_chrom17_list',17: 'gene_chrom18_list',18: 'gene_chrom19_list',
	19: 'gene_chrom20_list',20: 'gene_chrom21_list',21: 'gene_chrom22_list',22: 'gene_chromX_list',23: 'gene_chromY_list'}

	# open each chrom file for ACC,Chromosome 0-23
	for chrom in range(len(chromosome_directories)):
		gene_name_list = list()
		with open(chromosome_directories[chrom]) as mf:
			for line in mf:
				line = line.strip()
				cols = line.split()
				gene_name_list.append(cols[0])
		gene_chrom_dict[chrom] = gene_name_list
		# print(f"gene_chrom_dict[{chrom}]:", gene_chrom_dict[chrom])


		# print("gene_chrom_dict[",chrom,"]=",len(gene_chrom_dict[chrom]))
				
	# Build a gene dictionary for each chromosome
	gene_index_dict = {0: 'gene_index_chrom01',1: 'gene_index_chrom02',2: 'gene_index_chrom03',
	3: 'gene_index_chrom04',4: 'gene_index_chrom05',5: 'gene_index_chrom06',6: 'gene_index_chrom07',
	7: 'gene_index_chrom08',8: 'gene_index_chrom09',9: 'gene_index_chrom10',10: 'gene_index_chrom11',
	11: 'gene_index_chrom12',12: 'gene_index_chrom13',13: 'gene_index_chrom14',14: 'gene_index_chrom15',
	15: 'gene_index_chrom16',16: 'gene_index_chrom17',17: 'gene_index_chrom18',18: 'gene_index_chrom19',
	19: 'gene_index_chrom20',20: 'gene_index_chrom21',21: 'gene_index_chrom22',22: 'gene_index_chromX',23: 'gene_index_chromY'}
	for chrom in range(len(chromosome_directories)):
		# print("chrom is ",chrom) # 0-23
		gene_dict = dict()
		gene_index = 0
		for gene in range(len(gene_chrom_dict[chrom])):
			gene_dict[gene_chrom_dict[chrom][gene]] = gene_index
			gene_index += 1
		gene_index_dict[chrom] = gene_dict # genename:geneindex
		# print(f"gene_index_dic[{chrom}]:",gene_index_dict[chrom])
	# for i in range(len(chromosome_directories)):
	# 	print(f"len[{i}]",len(gene_index_dict[i]))
	# Create an array for each chromosome according to the corresponding gene length
	gene_array_dict = {0: 'gene_array_chrom01',1: 'gene_array_chrom02',2: 'gene_array_chrom03',
	3: 'gene_array_chrom04',4: 'gene_array_chrom05',5: 'gene_array_chrom06',6: 'gene_array_chrom07',
	7: 'gene_array_chrom08',8: 'gene_array_chrom09',9: 'gene_array_chrom10',10: 'gene_array_chrom11',
	11: 'gene_array_chrom12',12: 'gene_array_chrom13',13: 'gene_array_chrom14',14: 'gene_array_chrom15',
	15: 'gene_array_chrom16',16: 'gene_array_chrom17',17: 'gene_array_chrom18',18: 'gene_array_chrom19',
	19: 'gene_array_chrom20',20: 'gene_array_chrom21',21: 'gene_array_chrom22',22: 'gene_array_chromX',23: 'gene_array_chromY'}
	for chrom in range(len(chromosome_directories)):
		geneNum = len(gene_chrom_dict[chrom])
		geneWidth = geneNum//310
		if geneNum%310!=0:
			geneWidth += 1 # each chrom occupies geneWidth*3
		gene_array_dict[chrom] = np.zeros([310, geneWidth*3, 3], np.uint8) + 255
	# print("gene_array_dict[0]",gene_array_dict[0])
	############## Read sample file #################
	os.chdir(cancer_sample_directory_path)
	# print("cancer_sample_directory_path=",cancer_sample_directory_path)
	file_name_list = list()
	A_sample_id_list = []
	A_sample_cancer_list = []
	A_sample_name_list = []

	with open('MANIFEST.txt', 'r') as mf:
		for line in mf:
			line = line.strip()
			cols = line.split()
			A_sample_id_list=cols[0]
			file_name_list.append(cols[1].split('/')[1].split('.gz')[0])
	# print("file_name_list",file_name_list)
	fig_path = "../mydataset/"

	for i in range(len(file_name_list)):
		A_sample_cancer_list.append(cancer_name)
		A_sample_name_list.append(file_name_list[i])
		if not os.path.exists(fig_path+A_sample_cancer_list[i]):
			os.makedirs(fig_path+A_sample_cancer_list[i])
	# Iterate over each sample
	for i in range(0, len(file_name_list)):
		# sample_id = A_sample_id_list[i]
		cancer_id = A_sample_cancer_list[i]
		sample_file = A_sample_name_list[i]
		file_name = sample_file + "_" + cancer_id
		sample_path = cancer_sample_directory_path + "/" + sample_file # Sample path
		# print("sample_path:",sample_path)
		data = np.zeros([310, 310, 3], np.uint8) + 255 # Create a three-dimensional matrix

		if os.path.exists(sample_path):
			begin = datetime.datetime.now()
			with open(sample_path, 'r', encoding='ISO-8859-1') as f:
				for _ in range(7):
					next(f)
				header = next(f).strip().split('\t')
				hugo_symbol_index = header.index('Hugo_Symbol')
				variant_type_index = header.index('Variant_Type')
				chromosome_index = header.index('Chromosome')
				cling_sig_index = header.index("CLIN_SIG")
				somatic_index=header.index("SOMATIC")
				# print(hugo_symbol_index,variant_type_index,chromosome_index,somatic_index)
				# test
				snpNum = 0
				insNum = 0
				delNum = 0
				for line in f:
					line = line.strip()
					cols = line.split("\t")
					gene = cols[hugo_symbol_index]
					variant = cols[variant_type_index]
					chrom = cols[chromosome_index]
					somatic=cols[somatic_index]
					cling_sig=cols[cling_sig_index]
					# print(gene,variant,chrom)
					if chrom == "chr1" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						# The index of this gene on chromosome 1
						gene_index = gene_index_dict[0][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[0][idx,jdx] = [255,0,0]
							if(somatic=="1;1" or somatic=='1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[0][idx,jdx], increase_factor)
								gene_array_dict[0][idx, jdx] = new_color
							snpNum += 1
						elif(variant == 'INS'):
							gene_array_dict[0][idx,jdx+1] = [0,255,0]
							if(cling_sig=='pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[0][idx, jdx+1], increase_factor)
								gene_array_dict[0][idx, jdx+1] = new_color
							insNum += 1
						elif(variant == 'DEL'):
							gene_array_dict[0][idx,jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[0][idx, jdx + 2], increase_factor)
								gene_array_dict[0][idx, jdx + 2] = new_color
							delNum += 1
					elif chrom == "chr2" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						# The index of this gene on chromosome 2
						gene_index = gene_index_dict[1][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[1][idx, jdx] = [255,0,0]
							if (somatic == "1;1" or somatic == '1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[1][idx, jdx], increase_factor)
								gene_array_dict[1][idx, jdx] = new_color
							snpNum += 1
						elif(variant == 'INS'):
							gene_array_dict[1][idx, jdx+1] = [0,255,0]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[1][idx, jdx + 1], increase_factor)
								gene_array_dict[1][idx, jdx + 1] = new_color
							insNum += 1
						elif(variant == 'DEL'):
							gene_array_dict[1][idx, jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[1][idx, jdx + 2], increase_factor)
								gene_array_dict[1][idx, jdx + 2] = new_color
							delNum += 1
					elif chrom == "chr3" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						gene_index = gene_index_dict[2][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[2][idx, jdx] = [255,0,0]
							if (somatic == "1;1" or somatic == '1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[2][idx, jdx], increase_factor)
								gene_array_dict[2][idx, jdx] = new_color
							snpNum += 1
						elif(variant == 'INS'):
							gene_array_dict[2][idx, jdx+1] = [0,255,0]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[2][idx, jdx + 1], increase_factor)
								gene_array_dict[2][idx, jdx + 1] = new_color
							insNum += 1
						elif(variant == 'DEL'):
							gene_array_dict[2][idx, jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[2][idx, jdx + 2], increase_factor)
								gene_array_dict[2][idx, jdx + 2] = new_color
							delNum += 1
					elif chrom == "chr4" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						gene_index = gene_index_dict[3][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[3][idx, jdx] = [255,0,0]
							if (somatic == "1;1" or somatic == '1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[3][idx, jdx], increase_factor)
								gene_array_dict[3][idx, jdx] = new_color
							snpNum += 1
						elif(variant == 'INS'):
							gene_array_dict[3][idx, jdx+1] = [0,255,0]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[3][idx, jdx + 1], increase_factor)
								gene_array_dict[3][idx, jdx + 1] = new_color
							insNum += 1
						elif(variant == 'DEL'):
							gene_array_dict[3][idx, jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[3][idx, jdx + 2], increase_factor)
								gene_array_dict[3][idx, jdx + 2] = new_color
							delNum += 1
					elif chrom == "chr5" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						gene_index = gene_index_dict[4][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[4][idx, jdx] = [255,0,0]
							if (somatic == "1;1" or somatic == '1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[4][idx, jdx], increase_factor)
								gene_array_dict[4][idx, jdx] = new_color
							snpNum += 1
						elif(variant == 'INS'):
							gene_array_dict[4][idx, jdx+1] = [0,255,0]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[4][idx, jdx + 1], increase_factor)
								gene_array_dict[4][idx, jdx + 1] = new_color
							insNum += 1
						elif(variant == 'DEL'):
							gene_array_dict[4][idx, jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[4][idx, jdx + 2], increase_factor)
								gene_array_dict[4][idx, jdx + 2] = new_color
							delNum += 1
					elif chrom == "chr6" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						gene_index = gene_index_dict[5][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[5][idx, jdx] = [255,0,0]
							if (somatic == "1;1" or somatic == '1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[5][idx, jdx], increase_factor)
								gene_array_dict[5][idx, jdx] = new_color
							snpNum += 1
						elif(variant == 'INS'):
							gene_array_dict[5][idx, jdx+1] = [0,255,0]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[5][idx, jdx + 1], increase_factor)
								gene_array_dict[5][idx, jdx + 1] = new_color
							insNum += 1
						elif(variant == 'DEL'):
							gene_array_dict[5][idx, jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[5][idx, jdx + 2], increase_factor)
								gene_array_dict[5][idx, jdx + 2] = new_color
							delNum += 1
					elif chrom == "chr7" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						gene_index = gene_index_dict[6][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[6][idx, jdx] = [255,0,0]
							if (somatic == "1;1" or somatic == '1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[6][idx, jdx], increase_factor)
								gene_array_dict[6][idx, jdx] = new_color
							snpNum += 1
						elif(variant == 'INS'):
							gene_array_dict[6][idx, jdx+1] = [0,255,0]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[6][idx, jdx + 1], increase_factor)
								gene_array_dict[6][idx, jdx + 1] = new_color
							insNum += 1
						elif(variant == 'DEL'):
							gene_array_dict[6][idx, jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[6][idx, jdx + 2], increase_factor)
								gene_array_dict[6][idx, jdx + 2] = new_color
							delNum += 1
					elif chrom == "chr8" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						gene_index = gene_index_dict[7][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[7][idx, jdx] = [255,0,0]
							if (somatic == "1;1" or somatic == '1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[7][idx, jdx], increase_factor)
								gene_array_dict[7][idx, jdx] = new_color
							snpNum += 1
						elif(variant == 'INS'):
							gene_array_dict[7][idx, jdx+1] = [0,255,0]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[7][idx, jdx + 1], increase_factor)
								gene_array_dict[7][idx, jdx + 1] = new_color
							insNum += 1
						elif(variant == 'DEL'):
							gene_array_dict[7][idx, jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[7][idx, jdx + 2], increase_factor)
								gene_array_dict[7][idx, jdx + 2] = new_color
							delNum += 1
					elif chrom == "chr9" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						gene_index = gene_index_dict[8][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[8][idx, jdx] = [255,0,0]
							if (somatic == "1;1" or somatic == '1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[8][idx, jdx], increase_factor)
								gene_array_dict[8][idx, jdx] = new_color
							snpNum += 1
						elif(variant == 'INS'):
							gene_array_dict[8][idx, jdx+1] = [0,255,0]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[8][idx, jdx + 1], increase_factor)
								gene_array_dict[8][idx, jdx + 1] = new_color
							insNum += 1
						elif(variant == 'DEL'):
							gene_array_dict[8][idx, jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[8][idx, jdx + 2], increase_factor)
								gene_array_dict[8][idx, jdx + 2] = new_color
							delNum += 1
					elif chrom == "chr10" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						gene_index = gene_index_dict[9][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[9][idx, jdx] = [255,0,0]
							if (somatic == "1;1" or somatic == '1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[9][idx, jdx], increase_factor)
								gene_array_dict[9][idx, jdx] = new_color
							snpNum += 1
						elif(variant == 'INS'):
							gene_array_dict[9][idx, jdx+1] = [0,255,0]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[9][idx, jdx + 1], increase_factor)
								gene_array_dict[9][idx, jdx + 1] = new_color
							insNum += 1
						elif(variant == 'DEL'):
							gene_array_dict[9][idx, jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[9][idx, jdx + 2], increase_factor)
								gene_array_dict[9][idx, jdx + 2] = new_color
							delNum += 1
					elif chrom == "chr11" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						gene_index = gene_index_dict[10][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[10][idx, jdx] = [255,0,0]
							if (somatic == "1;1" or somatic == '1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[10][idx, jdx], increase_factor)
								gene_array_dict[10][idx, jdx] = new_color
							snpNum += 1
						elif(variant == 'INS'):
							gene_array_dict[10][idx, jdx+1] = [0,255,0]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[10][idx, jdx + 1], increase_factor)
								gene_array_dict[10][idx, jdx + 1] = new_color
							insNum += 1
						elif(variant == 'DEL'):
							gene_array_dict[10][idx, jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[10][idx, jdx + 2], increase_factor)
								gene_array_dict[10][idx, jdx + 2] = new_color
							delNum += 1
					elif chrom == "chr12" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						gene_index = gene_index_dict[11][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[11][idx, jdx] = [255,0,0]
							if (somatic == "1;1" or somatic == '1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[11][idx, jdx], increase_factor)
								gene_array_dict[11][idx, jdx] = new_color
							snpNum += 1
						elif(variant == 'INS'):
							gene_array_dict[11][idx, jdx+1] = [0,255,0]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[11][idx, jdx + 1], increase_factor)
								gene_array_dict[11][idx, jdx + 1] = new_color
							insNum += 1
						elif(variant == 'DEL'):
							gene_array_dict[11][idx, jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[11][idx, jdx + 2], increase_factor)
								gene_array_dict[11][idx, jdx + 2] = new_color
							delNum += 1
					elif chrom == "chr13" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						gene_index = gene_index_dict[12][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[12][idx, jdx] = [255,0,0]
							if (somatic == "1;1" or somatic == '1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[12][idx, jdx], increase_factor)
								gene_array_dict[12][idx, jdx] = new_color
							snpNum += 1
						elif(variant == 'INS'):
							gene_array_dict[12][idx, jdx+1] = [0,255,0]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[12][idx, jdx + 1], increase_factor)
								gene_array_dict[12][idx, jdx + 1] = new_color
							insNum += 1
						elif(variant == 'DEL'):
							gene_array_dict[12][idx, jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[12][idx, jdx + 2], increase_factor)
								gene_array_dict[12][idx, jdx + 2] = new_color
							delNum += 1
					elif chrom == "chr14" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						gene_index = gene_index_dict[13][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[13][idx, jdx] = [255,0,0]
							if (somatic == "1;1" or somatic == '1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[13][idx, jdx], increase_factor)
								gene_array_dict[13][idx, jdx] = new_color
							snpNum += 1
						elif(variant == 'INS'):
							gene_array_dict[13][idx, jdx+1] = [0,255,0]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[13][idx, jdx + 1], increase_factor)
								gene_array_dict[13][idx, jdx + 1] = new_color
							insNum += 1
						elif(variant == 'DEL'):
							gene_array_dict[13][idx, jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[13][idx, jdx + 2], increase_factor)
								gene_array_dict[13][idx, jdx + 2] = new_color
							delNum += 1
					elif chrom == "chr15" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						gene_index = gene_index_dict[14][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[14][idx, jdx] = [255,0,0]
							if (somatic == "1;1" or somatic == '1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[14][idx, jdx], increase_factor)
								gene_array_dict[14][idx, jdx] = new_color
							snpNum += 1
						elif(variant == 'INS'):
							gene_array_dict[14][idx, jdx+1] = [0,255,0]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[14][idx, jdx + 1], increase_factor)
								gene_array_dict[14][idx, jdx + 1] = new_color
							insNum += 1
						elif(variant == 'DEL'):
							gene_array_dict[14][idx, jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[14][idx, jdx + 2], increase_factor)
								gene_array_dict[14][idx, jdx + 2] = new_color
							delNum += 1
					elif chrom == "chr16" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						gene_index = gene_index_dict[15][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[15][idx, jdx] = [255,0,0]
							if (somatic == "1;1" or somatic == '1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[15][idx, jdx], increase_factor)
								gene_array_dict[15][idx, jdx] = new_color
							snpNum += 1
						elif(variant == 'INS'):
							gene_array_dict[15][idx, jdx+1] = [0,255,0]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[15][idx, jdx + 1], increase_factor)
								gene_array_dict[15][idx, jdx + 1] = new_color
							insNum += 1
						elif(variant == 'DEL'):
							gene_array_dict[15][idx, jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[15][idx, jdx + 2], increase_factor)
								gene_array_dict[15][idx, jdx + 2] = new_color
							delNum += 1
					elif chrom == "chr17" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						gene_index = gene_index_dict[16][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[16][idx, jdx] = [255,0,0]
							if (somatic == "1;1" or somatic == '1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[16][idx, jdx], increase_factor)
								gene_array_dict[16][idx, jdx] = new_color
							snpNum += 1
						elif(variant == 'INS'):
							gene_array_dict[16][idx, jdx+1] = [0,255,0]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[16][idx, jdx + 1], increase_factor)
								gene_array_dict[16][idx, jdx + 1] = new_color
							insNum += 1
						elif(variant == 'DEL'):
							gene_array_dict[16][idx, jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[16][idx, jdx + 2], increase_factor)
								gene_array_dict[16][idx, jdx + 2] = new_color
							delNum += 1
					elif chrom == "chr18" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						gene_index = gene_index_dict[17][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[17][idx, jdx] = [255,0,0]
							if (somatic == "1;1" or somatic == '1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[17][idx, jdx], increase_factor)
								gene_array_dict[17][idx, jdx] = new_color
							snpNum += 1
						elif(variant == 'INS'):
							gene_array_dict[17][idx, jdx+1] = [0,255,0]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[17][idx, jdx + 1], increase_factor)
								gene_array_dict[17][idx, jdx + 1] = new_color
							insNum += 1
						elif(variant == 'DEL'):
							gene_array_dict[17][idx, jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[17][idx, jdx + 2], increase_factor)
								gene_array_dict[17][idx, jdx + 2] = new_color
							delNum += 1
					elif chrom == "chr19" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						gene_index = gene_index_dict[18][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[18][idx, jdx] = [255,0,0]
							if (somatic == "1;1" or somatic == '1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[18][idx, jdx], increase_factor)
								gene_array_dict[18][idx, jdx] = new_color
							snpNum += 1
						elif(variant == 'INS'):
							gene_array_dict[18][idx, jdx+1] = [0,255,0]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[18][idx, jdx + 1], increase_factor)
								gene_array_dict[18][idx, jdx + 1] = new_color
							insNum += 1
						elif(variant == 'DEL'):
							gene_array_dict[18][idx, jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[18][idx, jdx + 2], increase_factor)
								gene_array_dict[18][idx, jdx + 2] = new_color
							delNum += 1
					elif chrom == "chr20" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						gene_index = gene_index_dict[19][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[19][idx, jdx] = [255,0,0]
							if (somatic == "1;1" or somatic == '1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[19][idx, jdx], increase_factor)
								gene_array_dict[19][idx, jdx] = new_color
							snpNum += 1
						elif(variant == 'INS'):
							gene_array_dict[19][idx, jdx+1] = [0,255,0]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[19][idx, jdx + 1], increase_factor)
								gene_array_dict[19][idx, jdx + 1] = new_color
							insNum += 1
						elif(variant == 'DEL'):
							gene_array_dict[19][idx, jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[19][idx, jdx + 2], increase_factor)
								gene_array_dict[19][idx, jdx + 2] = new_color
							delNum += 1
					elif chrom == "chr21" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						gene_index = gene_index_dict[20][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[20][idx, jdx] = [255,0,0]
							if (somatic == "1;1" or somatic == '1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[20][idx, jdx], increase_factor)
								gene_array_dict[20][idx, jdx] = new_color
							snpNum += 1
						elif(variant == 'INS'):
							gene_array_dict[20][idx, jdx+1] = [0,255,0]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[20][idx, jdx + 1], increase_factor)
								gene_array_dict[20][idx, jdx + 1] = new_color
							insNum += 1
						elif(variant == 'DEL'):
							gene_array_dict[20][idx, jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[20][idx, jdx + 2], increase_factor)
								gene_array_dict[20][idx, jdx + 2] = new_color
							delNum += 1
					elif chrom == "chr22" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						gene_index = gene_index_dict[21][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[21][idx, jdx] = [255,0,0]
							if (somatic == "1;1" or somatic == '1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[21][idx, jdx], increase_factor)
								gene_array_dict[21][idx, jdx] = new_color
							snpNum += 1
						elif(variant == 'INS'):
							gene_array_dict[21][idx, jdx+1] = [0,255,0]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[21][idx, jdx + 1], increase_factor)
								gene_array_dict[21][idx, jdx + 1] = new_color
							insNum += 1
						elif(variant == 'DEL'):
							gene_array_dict[21][idx, jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[21][idx, jdx + 2], increase_factor)
								gene_array_dict[21][idx, jdx + 2] = new_color
							delNum += 1
					elif chrom == "chrX" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						gene_index = gene_index_dict[22][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[22][idx, jdx] = [255,0,0]
							if (somatic == "1;1" or somatic == '1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[22][idx, jdx], increase_factor)
								gene_array_dict[22][idx, jdx] = new_color
							snpNum += 1
						elif(variant == 'INS'):
							gene_array_dict[22][idx, jdx+1] = [0,255,0]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[22][idx, jdx + 1], increase_factor)
								gene_array_dict[22][idx, jdx + 1] = new_color
							insNum += 1
						elif(variant == 'DEL'):
							gene_array_dict[22][idx, jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[22][idx, jdx + 2], increase_factor)
								gene_array_dict[22][idx, jdx + 2] = new_color
							delNum += 1
					elif chrom == "chrY" and (variant == 'SNP' or variant == 'INS' or variant == 'DEL'):
						gene_index = gene_index_dict[23][gene] #The real position, because the array starts from 0
						if gene_index < 310:
							idx = gene_index
							jdx = 0
						else:
							term = gene_index//310
							idx = gene_index%310
							jdx = term * 3
						if(variant == 'SNP'):
							gene_array_dict[23][idx, jdx] = [255,0,0]
							if (somatic == "1;1" or somatic == '1;1;1'):
								increase_factor = 1.2
								new_color = increase_brightness(gene_array_dict[23][idx, jdx], increase_factor)
								gene_array_dict[23][idx, jdx] = new_color
							snpNum += 1
							# print("snpNum=",snpNum)
						elif(variant == 'INS'):
							gene_array_dict[23][idx, jdx+1] = [0,255,0]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[23][idx, jdx + 1], increase_factor)
								gene_array_dict[23][idx, jdx + 1] = new_color
							insNum += 1
							# print("insNum=",insNum)
						elif(variant == 'DEL'):
							gene_array_dict[23][idx, jdx+2] = [0,0,255]
							if (cling_sig == 'pathogenic'):
								increase_factor = 1.5
								new_color = increase_brightness(gene_array_dict[23][idx, jdx + 2], increase_factor)
								gene_array_dict[23][idx, jdx + 2] = new_color
							delNum += 1
							# print("delNum=",delNum)
			p = 0 # p, q mark the beginning and end of the column position
			for chrom in range(len(chromosome_directories)): # 24 chromosome
				
				geneNum = len(gene_chrom_dict[chrom])

				q = (geneNum//310)*3
				if geneNum%310!=0:
					q = q + 3

				if p+q <= 310:
					data[:,p:(p+q)] = gene_array_dict[chrom]
					p = p+q

				else:
					print("The number of genes is over 310 lines!")
					print("cancer_id=",cancer_id)
					print("chrom=",chrom+1)
					print("data.shape=",data[:,p:(p+q)].shape)
				
		# save mutation map
		img_path = fig_path + cancer_name + '/' + file_name + ".png"
		cv2.imwrite(img_path, data)
		# print("Write img:", img_path)
		end = datetime.datetime.now()

def main():
	current_path = os.getcwd()
	#cancer_directories = [name for name in os.listdir(".") if os.path.isdir(name)]
	#cancer_directories.remove('ChromosomeGene')
	cancer_directories = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC',
						  'KIRP', 'LAML','LGG','LIHC','LUAD','LUSC','MESO','OV','PAAD','PCPG','PRAD','READ','SARC','SKCM','STAD',
						  'TGCT','THCA','THYM','UCEC','UCS','UVM']
	print("cancer_directories=", cancer_directories)

	for cancer in range(len(cancer_directories)):
		cancer_name = cancer_directories[cancer]
		print("cancer_name", cancer_name)
		data_chrom_path = current_path + "/gene_counts/all_diseases_genes/"  # Chromosome file path
		cancer_sample_directory_path = os.path.join(current_path, cancer_name)  # Sample path
		read_gene_names(data_chrom_path, cancer_sample_directory_path, cancer_name)
		os.chdir(current_path)

if __name__ == "__main__":
	main()
