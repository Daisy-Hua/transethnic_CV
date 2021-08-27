#!/usr/bin/env bash

#echo "Enter the target GENE ID :(e.g. ENSG00000167074.9)"
#read gene
#echo "Enter the gene name of this ID $gene"
#read gene_name
#echo "Your target gene is $gene_name, its ID is $gene"

# find the target SNPs for your target gene,the output file_preparation.sh's path is data/target/${gene_name}_SNP.txt.
python code/D_solver/SNP_finder.py $1 $2

#create genotype matrix
cd data/clean
mkdir ${2}_genotype
#eur genotype matrix
vcftools --vcf ../GEUVADIS.chr22.PH1PH2_465.IMPFRQFILT_BIALLELIC_PH.annotv2.genotypes.vcf --out ${2}_genotype/eur_${2}_genotype --chr 22 --snps ../target/${2}_SNP.txt --keep eur_genotype.012.indv --012
#afr genotype matrix
vcftools --vcf ../GEUVADIS.chr22.PH1PH2_465.IMPFRQFILT_BIALLELIC_PH.annotv2.genotypes.vcf --out ${2}_genotype/afr_${2}_genotype --chr 22 --snps ../target/${2}_SNP.txt --keep afr_genotype.012.indv --012

cd ../..

python code/D_solver/CV_model.py $1 $2 >> result/CV_model.txt

echo "$2 Done"