import pandas as pd
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path, enet_path

gene = sys.argv[1]
gene_name = sys.argv[2]

pheno_total = pd.read_csv('data/GD462.GeneQuantRPKM.50FN.samplename.resk10.txt', sep = '\t', index_col = 'TargetID')
target_pheno_total = pheno_total.loc[gene]

eur_sample = pd.read_csv('data/clean/'+gene_name+'_genotype/eur_'+gene_name+'_genotype.012.indv', sep = '\t', header = None)
afr_sample = pd.read_csv('data/clean/'+gene_name+'_genotype/afr_'+gene_name+'_genotype.012.indv', sep = '\t',header = None)

target_pheno_eur = pd.merge(target_pheno_total, eur_sample, left_index = True, right_on = 0)
target_pheno_afr = pd.merge(target_pheno_total, afr_sample, left_index = True, right_on = 0)

eur_genotype = pd.read_csv('data/clean/'+gene_name+'_genotype/eur_'+gene_name+'_genotype.012', sep = '\t', header = None, index_col = 0)
afr_genotype = pd.read_csv('data/clean/'+gene_name+'_genotype/afr_'+gene_name+'_genotype.012', sep = '\t', header = None, index_col = 0)

pa = target_pheno_afr.set_index(0)
pe = target_pheno_eur.set_index(0)

#make sure the genotype matrix's order is the same as that of phenotype vector
sorted_eur_pheno = pd.merge(eur_sample, pe, left_on = 0 , right_index = True, how = 'left')
sorted_afr_pheno = pd.merge(afr_sample, pa, left_on = 0 , right_index = True, how = 'left')

#original matrix(before standardization)
X1o = eur_genotype
X2o = afr_genotype
y1o = sorted_eur_pheno[gene]
y2o = sorted_afr_pheno[gene]

# least squares for marginal effect sizes
X1o = np.array(X1o,dtype = np.float64,order = 'C')
X2o = np.array(X2o,dtype = np.float64,order = 'C')
y1o = np.array(y1o,dtype = np.float64,order = 'C')
y2o = np.array(y2o,dtype = np.float64,order = 'C')

b1_hat = []
count3 = 0
for x1 in X1o.T:
    count3+=1
    x1 = np.vstack([x1, np.ones(len(x1))]).T
    b_hat, intercept = np.linalg.lstsq(x1,y1o)[0]
    b1_hat.append(b_hat)
print(count3)
b1_hat = np.array(b1_hat, dtype = np.float64, order = 'C')

plt.hist(b1_hat, bins = int(count3/4))
plt.savefig('result/'+gene_name+'_bhat_hist.png')