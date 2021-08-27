import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

#copy below codes to Console might be easier for following calculation
pheno_total = pd.read_csv('data/GD462.GeneQuantRPKM.50FN.samplename.resk10.txt', sep = '\t', index_col = 'TargetID')
target_pheno_total = pheno_total.loc['ENSG00000167074.9']

b_hat_eur_total = pd.read_csv('data/EUR373.gene.cis.FDR5.all.rs137.txt', sep = '\t')
b_hat_eur_total_grouped = dict(list(b_hat_eur_total.groupby('GENE_ID')))
b_hat_eur_target = b_hat_eur_total_grouped['ENSG00000167074.9']

eur_sample = pd.read_csv('data/clean/eur_genotype.012.indv', sep = '\t', header = None)
afr_sample = pd.read_csv('data/clean/afr_genotype.012.indv', sep = '\t',header = None)

target_pheno_eur = pd.merge(target_pheno_total, eur_sample, left_index = True, right_on = 0)
target_pheno_afr = pd.merge(target_pheno_total, afr_sample, left_index = True, right_on = 0)

eur_genotype = pd.read_csv('data/clean/eur_genotype.012', sep = '\t', header = None, index_col = 0)
afr_genotype = pd.read_csv('data/clean/afr_genotype.012', sep = '\t', header = None, index_col = 0)

pa = target_pheno_afr.set_index(0)
pe = target_pheno_eur.set_index(0)

sorted_eur_pheno = pd.merge(eur_sample, pe, left_on = 0 , right_index = True, how = 'left')
sorted_afr_pheno = pd.merge(afr_sample, pa, left_on = 0 , right_index = True, how = 'left')
sorted_b_hat_eur = b_hat_eur_target.sort_values(by = 'SNPpos')

from scipy import optimize
X1 = eur_genotype-eur_genotype.mean()
X2 = afr_genotype-afr_genotype.mean()
b1 = sorted_b_hat_eur['rvalue']
y1 = sorted_eur_pheno['ENSG00000167074.9']-sorted_eur_pheno['ENSG00000167074.9'].mean()
y2 = sorted_afr_pheno['ENSG00000167074.9']-sorted_afr_pheno['ENSG00000167074.9'].mean()
N1 = 373 #number of eur samples
R1 = X1.cov()

def lasso_fun(beta):
    betaT_R1 = np.dot(beta.T, R1)
    D_R1 = np.diag(np.diag(R1))
    b1T_D_R1 = np.dot(b1.T, D_R1)
    term1 = (N1-1)*(np.dot(betaT_R1, beta) - 2 * np.dot(b1T_D_R1, beta))
    term2 = np.linalg.norm(y2 - np.dot(X2, beta))**2 #L2 norm should be squared here(least square)
    penalty = np.linalg.norm(beta, ord = 1)
    return term1 + term2 + penalty

beta = b1 #guessing best vector for the optimize function?

#regard eur and afr population as one giant population, using individual level data (both eur and afr) as input
#integreted_eur_afr = pd.concat([eur_genotype, afr_genotype])
#integereted_pheno = pd.concat([sorted_eur_pheno, sorted_afr_pheno])
#integereted_pheno = integereted_pheno.set_index(0)
#X12 = integreted_eur_afr-integreted_eur_afr.mean()
#y12 = integereted_pheno-integereted_pheno.mean()

result = optimize.minimize(lasso_fun, beta, maxiter = 100)

