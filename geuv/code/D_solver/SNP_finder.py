import sys
import pandas as pd

gene = sys.argv[1]
gene_name = sys.argv[2]

pheno_total = pd.read_csv('data/GD462.GeneQuantRPKM.50FN.samplename.resk10.txt', sep = '\t', index_col = 'TargetID')
b_hat_eur_total = pd.read_csv('data/EUR373.gene.cis.FDR5.all.rs137.txt', sep = '\t')
b_hat_eur_total_grouped = dict(list(b_hat_eur_total.groupby('CHR_SNP')))
SNP_22 = b_hat_eur_total_grouped[22]
SNP_22.to_csv('data/SNP_22.txt', sep = '\t')
SNP22_Gene = dict(list(SNP_22.groupby('GENE_ID')))
GENE = SNP22_Gene[gene]
eur_sample = pd.read_csv('data/clean/eur_genotype.012.indv', sep='\t', header=None)
afr_sample = pd.read_csv('data/clean/afr_genotype.012.indv', sep='\t', header=None)
snp_total = pd.read_csv('data/Phase1.Geuvadis_dbSnp137_idconvert.txt', sep = '\t', header = None)
GENE_SNP = GENE['SNP_ID']
GENE_SNP_valid = pd.merge(snp_total,GENE,left_on = 0, right_on = 'SNP_ID')
GENE_SNP_valid[1].to_csv('data/target/'+gene_name+'_SNP.txt', sep = '\t',index = False, header = False)