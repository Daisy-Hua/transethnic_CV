import sys
import pandas as pd

path = sys.argv[1]
type = sys.argv[2]

mix = pd.read_csv(path, sep = '\t', header = None)
mix = mix.transpose()
mix = mix.rename(columns = mix.iloc[2])

refer = pd.read_csv('data/igsr_samples.tsv', sep = '\t')
refer = refer[["Sample name","Superpopulation code"]]
split = pd.merge(refer, mix, left_on ='Sample name', right_on = 'ID')
split = split.drop(['Sample name'], axis = 1)
split = split.set_index(['Superpopulation code'])

eur = split.loc['EUR']
afr = split.loc['AFR']

if type == 'geno':
    eur.to_csv('data/eurgeno.txt', sep = '\t')
    afr.to_csv('data/afrgeno.txt', sep = '\t')

if type == 'pheno':
    eur.to_csv('data/eurpheno.txt', sep = '\t')
    afr.to_csv('data/afrpheno.txt', sep = '\t')
