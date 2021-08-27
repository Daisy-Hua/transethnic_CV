import sys
sys.path.append('code/transethnic_prs-main/')
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import transethnic_prs.model1.Model1Blk as model1blk
from scipy import optimize

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

#centralize
X1 = (X1o-X1o.mean())
X2 = (X2o-X2o.mean())
y1 = (y1o-y1o.mean())
y2 = (y2o-y2o.mean())
N1 = 373# number of eur samples
N2 = 89# number of afr samples
all_SNP = X1.shape[1]

#examine whether there are SNPs without variance (both in eur ane afr geno matrixes)
count1 = 0
idx_valid1 = []
for idx1,x1 in enumerate(X1.std()):
    if x1==0:
        count1 +=1
    else:
        idx_valid1.append(idx1)
idx_valid2 = []
count2 = 0
for idx2,x2 in enumerate(X2.std()):
    if x2==0:
        count2 +=1
    else:
        idx_valid2.append(idx2)
print(count1)
print(count2)

if count1 != 0 and count2 == 0:
    print("only Eur has %d invalid SNPs" % count1)
    idx_valid = idx_valid1
if count2 != 0 and count1 == 0:
    print("only Afr has %d invalid SNPs" % count2)
    idx_valid = idx_valid2
if count1 == 0 and count2 == 0:
    print('either of Eur or Afr has invalid SNPs')
    idx_valid = list(range(all_SNP))
if count1 != 0 and count2 != 0:
    print("Eur has %d invalid SNPs, " % count1, "Afr has %d invalid SNPs" % count2)
    idx_valid = list(set(idx_valid1).intersection(set(idx_valid2)))


R1 = eur_genotype.cov()#this step must be done before turning all the DataFrame type matrixes and vectors into ndarrays

#transpose matrix into C order to improve the speed
X1 = np.array(X1,dtype = np.float64,order = 'C')
X1o = np.array(X1o,dtype = np.float64,order = 'C')
X2 = np.array(X2,dtype = np.float64,order = 'C')
X2o = np.array(X2o,dtype = np.float64,order = 'C')
y1 = np.array(y1, dtype = np.float64, order = 'C')
y1o = np.array(y1o,dtype = np.float64,order = 'C')
y2 = np.array(y2, dtype = np.float64, order = 'C')
y2o = np.array(y2o,dtype = np.float64,order = 'C')
R1 = np.array(R1,dtype = np.float64, order = 'C')

#chose valid subsets of genotype matrixes
X1 = X1[:, idx_valid]
X2 = X2[:, idx_valid]
X1o = X1o[:, idx_valid]
X2o = X2o[:, idx_valid]

#standardize
X1 = X1/np.std(X1o, axis = 0)
X2 = X2/np.std(X2o, axis = 0)
check = np.isnan(X2).any(axis=0) #see if there are 'nan' values in the X2
for x in check:
    if x == True:
        print('Error in standardization')

y1 = y1/y1o.std()
y2 = y2/y2o.std()

R1 = np.cov(X1.T)
D_R1 = np.diag(np.diag(R1))

A1 = X1.T @ X1
A1_t = (N1-1)*R1
b1 = X1.T @ y1

#make sure X1 and X2 are C order ndarray
X1 = np.array(X1, dtype = np.float64, order = 'C')
X2 = np.array(X2, dtype = np.float64, order = 'C')

#yanyu's solver
mod1 = model1blk.Model1Blk([A1], [b1], [X2], y2)

#en
l1_ratio = 0.1
t1 = time.time()
beta_mat_en, lambda_seq_en, niters_en, tols_en, convs_en = mod1.solve_path(alpha=l1_ratio)
print(f'Yanyu EN solver Run time = {time.time()-t1} s')

# lasso
t2 = time.time()
beta_mat_lasso, lambda_seq_lasso, niters_lasso, tols_lasso, convs_lasso = mod1.solve_path(alpha=1)
print(f'Yanyu lasso solver Run time = {time.time()-t2} s')

#scipy.optimize solver

beta = np.zeros(R1.shape[0])#guess an approximate true beta
#lasso:
def lasso_fun(beta):
    term1 = beta.T @ A1 @ beta - 2 * b1.T @ beta
    term2 = np.linalg.norm(y2 - X2 @ beta)**2 #L2 norm should be squared here(least square)
    penalty = np.linalg.norm(beta, ord = 1)
    return term1 + term2 + penalty

#en:
def EN(beta):
    term1 = beta.T @ A1 @ beta - 2 * b1.T @ beta
    term2 = np.linalg.norm(y2 - X2 @ beta)**2 #L2 norm should be squared here(least square)
    ridge = np.linalg.norm(beta)**2
    lasso = np.linalg.norm(beta, ord = 1)
    penalty = 0.1 * lasso + 0.9 * ridge
    return term1 + term2 + penalty

t3 = time.time()
result = optimize.minimize(lasso_fun, beta)
print(f'scipy.optimize solver lasso Run time = {time.time()-t3} s')

t4 = time.time()
result_EN = optimize.minimize(EN,beta)
print(f'scipy.optimize solver en Run time = {time.time()-t4} s')

#create fig for observation vs prediction
y2_hat_lasso = X2 @ beta_mat_lasso[:,-1:]#yanyu's solver
y2_hat_lasso_op = X2 @ result.x#scipy.optimize solver
y2_hat_en = X2 @ beta_mat_en[:,-1:]
y2_hat_en_op = X2 @ result_EN.x


plt.title("scatter polt for different solvers")
plt.xlabel('observation')
plt.ylabel('prediction')
scatter1 = plt.scatter(y2, y2_hat_lasso, color = "cornflowerblue")
scatter2 = plt.scatter(y2, y2_hat_lasso_op, color = "lightcoral")# red dots are scipy.optimize results
scatter3 = plt.scatter(y2, y2_hat_en, color = "blue")# blue dots are yanyu's solver results
scatter4 = plt.scatter(y2, y2_hat_en_op, color = "red")
line = plt.plot(y2, y2, color = 'k')
plt.legend([scatter1,scatter2,scatter3,scatter4],["yanyu's lasso","scipy's lasso","yanyu's EN","scipy's EN"])
plt.savefig('result/'+gene_name+'_plot.png')
