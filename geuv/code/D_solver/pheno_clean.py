import pandas as pd

pheno_total = pd.read_csv('data/GD462.ExonQuantCount.45N.50FN.samplename.resk10.txt', sep = '\t')
p_grouped = dict(list(pheno_total.groupby('chr')))
#chr 22 genes are divided into two subsets because some number of chromosome is char not int
chr22_1 = p_grouped[22]
chr22_2 = p_grouped['22']
#group by Gene ID ,eg, ENSG00000075234.12
chr22_1_grouped = dict(list(chr22_1.groupby('Gene_Symbol')))
chr22_2_grouped = dict(list(chr22_2.groupby('Gene_Symbol')))
target_gene = chr22_2_grouped['ENSG00000075234.12']