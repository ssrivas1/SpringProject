setwd("C:\\Users\\aghar\\Downloads\\msis-5223-deliverable-1-cobra-kai-main\\msis-5223-deliverable-1-cobra-kai-main\\data")


###########################################
#==============Read in data===============#
#Read in the data for both data sets.	#
###########################################

#==========================================
# Read in the data sample 
#==========================================

reduction_data1 = read.table("C:\\Users\\aghar\\Downloads\\msis-5223-deliverable-1-cobra-kai-main\\msis-5223-deliverable-1-cobra-kai-main\\data\\College_Data_Transformed.csv", quote = "",header=T, sep=",")

#Remove extra columns from the dataset and dividing it into two for PCA and FA
reduction_data = subset(reduction_data, select=-c(college_name,is_private))
reduc_data_pca <- reduction_data[1:382,]
reduc_data_fa <- reduction_data[382:764,]


#################################################
#==========Principal Component Analysis=========#
# Perform a PCA for PU, PEOU, and Intention;	#
# Best used for testing a new model using new	#
# theory; for established models, use Factor	#
# Analysis (see below).					#
# Assessment is based on the full variance in	#
# the data; Factor Analysis is based on the	#
# shared variance.					#
#################################################

reduc_data_pca=reduc_data_pca[complete.cases(reduc_data_pca),]
pcamodel_reduc = princomp(reduc_data_pca,cor=TRUE)		#save PCA model with loadings
pcamodel_reduc$sdev^2								#Only one component has an eigenvalue greater than 1.0

plot(pcamodel_reduc,main="Scree Plot")							#screeplot
biplot(pcamodel_reduc)								#biplot of PCA model; numbers are rows in data
#### Result: Screeplot indicates one component; Biplot shows the items are evenly distributed
#### Decision: 
library(Hmisc) 
library(corrplot)


cor(reduc_data_fa)

#################################################
#===============ML Factor Analysis==============#
# Perform a varimax rotation for all variables	#
# in the data. Use Factor Analysis for		#
# established models and theory. Assessment is	#
# based on shared variance within the model.	#
#################################################

#Dropping NA
reduc_data_fa=reduc_data_fa[complete.cases(reduc_data_fa),]

library(psych)
install.packages('GPArotation')
library(GPArotation)
fa(r=cor(reduc_data_fa1), nfactors=4, rotate="varimax", SMC=FALSE, fm="minres")
reduction_data.FA = factanal(reduc_data_fa1,factors=4,rotation="varimax", 
                             scores="none")					#run the factor analysis with 4 factors
reduction_data.FA						
