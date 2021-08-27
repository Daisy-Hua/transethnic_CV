import sys
sys.path.append('code/transethnic_prs-main/')
import pandas as pd

gene = sys.argv[1]
gene_name = sys.argv[2]

pheno_total = pd.read_csv('data/GD462.GeneQuantRPKM.50FN.samplename.resk10.txt', sep = '\t', index_col = 'TargetID')
target_pheno_total = pheno_total.loc[gene]

afr_sample = pd.read_csv('data/clean/'+gene_name+'_genotype/afr_'+gene_name+'_genotype.012.indv', sep = '\t',header = None)

target_pheno_afr = pd.merge(target_pheno_total, afr_sample, left_index = True, right_on = 0)

afr_genotype = pd.read_csv('data/clean/'+gene_name+'_genotype/afr_'+gene_name+'_genotype.012', sep = '\t', header = None, index_col = 0)

pa = target_pheno_afr.set_index(0)

#make sure the genotype matrix's order is the same as that of phenotype vector
sorted_afr_pheno = pd.merge(afr_sample, pa, left_on = 0 , right_index = True, how = 'left')

def check_nan(x):
    count = 0
    for x in x.describe().loc['min']:
        if x == -1:
            count+=1
    return count


# inpute nan with each SNP's mean value
f = lambda x : x.replace(-1,x.mean())
X = afr_genotype.apply(f,axis = 'index')
print(gene_name,check_nan(afr_genotype),check_nan(X))