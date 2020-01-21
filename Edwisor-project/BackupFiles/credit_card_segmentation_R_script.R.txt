rm(list = ls())

x = c('ggplot2','corrgram','randomForest','unbalanced','C50','dummies','e1071','Information',
      'rpart','gbm','ROSE','DMwR','caret')

setwd("C:/Users/jhemmann/Desktop/Edwisor-project/")

lapply(x,require,character.only = TRUE)

rm(x)

library(cluster)
library(factoextra)
library(dplR)
library(reshape2)
library(tidyverse) ## manipulating and visualizing data (plyr, purrr, ggplot2, knitr...)
library(kableExtra) ## make nice tables with wrapper for kable()
library(cluster)    ## clustering algorithms and gap statistic
library(GGally) ## create matrix of variable plots
library(NbClust) ## clustering algorithms and identification of best K
library(varhandle) ## to create dummies 
library(ggplot2)
library(animation)
library(tidyr)
library(RColorbrewer)
library(dplyr)

set.seed(25)

credit =  read.csv('credit-card-data.csv',header = T , na.strings = c(" ","","NA","NAN"))

str(credit)

summary(credit)

missing_val = data.frame(apply(credit, 2, function(x){sum(is.na(x))}))
missing_val

credit$CREDIT_LIMIT[is.na(credit$CREDIT_LIMIT)] = median(credit$CREDIT_LIMIT, na.rm = T)
credit$MINIMUM_PAYMENTS[is.na(credit$MINIMUM_PAYMENTS)] = median(credit$MINIMUM_PAYMENTS, na.rm = T)
sum(is.na(credit))

#New Variables creation# 
credit$Monthly_Avg_PURCHASES <- credit$PURCHASES/(credit$TENURE)
credit$Monthly_CASH_ADVANCE <- credit$CASH_ADVANCE/(credit$TENURE)
credit$LIMIT_USAGE <- credit$BALANCE/credit$CREDIT_LIMIT
credit$MIN_PAYMENTS_RATIO <- credit$PAYMENTS/credit$MINIMUM_PAYMENTS

credit$PURCHASE_TYPE <- dplyr::case_when(
  credit$ONEOFF_PURCHASES == 0 & credit$INSTALLMENTS_PURCHASES == 0 ~ 'none',
  credit$ONEOFF_PURCHASES > 0 & credit$INSTALLMENTS_PURCHASES == 0 ~ 'oneoff',
  credit$ONEOFF_PURCHASES > 0 & credit$INSTALLMENTS_PURCHASES > 0 ~ 'both_oneoff_installment',
  credit$ONEOFF_PURCHASES ==0 & credit$INSTALLMENTS_PURCHASES > 0 ~ 'installment'
)
credit$PURCHASE_TYPE  <- as.factor(credit$PURCHASE_TYPE)

colnames(credit)

str(credit)

credit_copy = credit

transformed_variables <- c("BALANCE","BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", 
                           "CASH_ADVANCE","PURCHASES_FREQUENCY","ONEOFF_PURCHASES_FREQUENCY","PURCHASES_INSTALLMENTS_FREQUENCY","CASH_ADVANCE_FREQUENCY", 
                           "CASH_ADVANCE_TRX", "PURCHASES_TRX", "CREDIT_LIMIT", 
                           "PAYMENTS", "MINIMUM_PAYMENTS","PRC_FULL_PAYMENT","TENURE",
                           "Monthly_Avg_PURCHASES","Monthly_CASH_ADVANCE","LIMIT_USAGE","MIN_PAYMENTS_RATIO") # vector of variables to be log transformed

credit_transformed_data <- credit %>%                           # preserve original dataset
  .[,-c(1,23)] %>%                                              #dropping CUST_ID and PURCHASE_TYPE
  mutate_at(vars(transformed_variables), funs(log(1 + .))) %>%  # add 1 to each value to avoid log(0)
              mutate_at(c(2:17), funs(c(scale(.))))                         # scale all numeric variables to mean of 0 & sd = 1
            
            
#taking only the variable which we use for KPI
credit_transformed_selected <- credit_transformed_data[,-c(1,3,6,17,14,15,16,13)]

colnames(credit_transformed_selected)

binary_purchase_type <- to.dummy(credit$PURCHASE_TYPE, "purchase_type")

credit_transformed_selected <- cbind(credit_transformed_selected,binary_purchase_type)

colnames(credit_transformed_selected)

#correlation map
cormat <- round(cor(credit_transformed_selected),3)
credit_corr <- melt(cormat)
ggplot(data = credit_corr,aes (x = Var1,y = Var2,fill = value)) +
theme(axis.text.x = element_text(angle = 45, vjust = 1,size = 12, hjust = 1)) +
geom_tile()

credit_scale = scale(credit_transformed_selected)

credit_scale = as.data.frame(credit_scale)
dim(credit_scale)

#PCA
credit_pca <- prcomp(credit_scale)

names(credit_pca)

credit_pca$center
credit_pca$scale
credit_pca$rotation
dim(credit_pca$x)

std_dev <- credit_pca$sdev
pca_var <- std_dev^2

pca_varex <- pca_var/sum(pca_var)
plot(pca_varex, xlab = "Principal Component",
   ylab = "Proportion of Variance Explained",
   type = "b")
#The plot above shows that ~ 5 components explains around 94% variance in the data set. 
#In order words, using PCA we have reduced 17 predictors to 5 without compromising on explained variance. 
#This is the power of PCA> Let's do a confirmation check, by plotting a cumulative variance plot. 
#This will give us a clear picture of number of components.

plot(cumsum(pca_varex), xlab = "Principal Component",
   ylab = "Cumulative Proportion of Variance Explained",
   type = "b")
#This plot shows that 5 components results in variance close to ~ 94%. 
#Therefore, in this case, we'll select number of components as 5 [PC1 to PC5] and proceed to the modeling stage. 
#This completes the steps to implement PCA on train data.
#For modeling, we'll use these 30 components as predictor variables and follow the normal procedures.

credit_pca_data <- data.frame(credit_pca$x)
credit_pca_data <- credit_pca_data[,1:5]
dim(credit_pca_data)

#kmeans.ani(credit_pca_data[1:2], 4)

#finding optimal K
kmean_withinss <- function(k) {
cluster <- kmeans(credit_pca_data, k)
return (cluster$tot.withinss)
}

max_k <- 10
wss <- sapply(2:max_k, kmean_withinss)

# Create a data frame to plot the graph
elbow <-data.frame(2:max_k, wss)

ggplot(elbow, aes(x = X2.max_k, y = wss)) +
geom_point() +
geom_line() +
scale_x_continuous(breaks = seq(1, 20, by = 1))

#From the graph, you can see the optimal k is 4, where the curve is starting to have a diminishing return.
#Once you have our optimal k, you re-run the algorithm with k equals to 4 and evaluate the clusters

pc_cluster_4 <-kmeans(credit_pca_data, 4)
cluster_df <- (pc_cluster_4$cluster)
pc_cluster_4$centers
pc_cluster_4$size

center <-pc_cluster_4$centers
center
# create dataset with the cluster number
cluster <- c(1: 4)
center_df <- data.frame(cluster, center)
# Reshape the data
center_reshape <- gather(center_df, features, values, PC1:PC5)
head(center_reshape)

cluster_scatter <- cbind(credit_pca_data,pc_cluster_4$cluster)
names(cluster_scatter)[6] <- "Cluster"
ggplot(data=cluster_scatter, aes(x=PC1, y=PC2, color=factor(Cluster))) + 
geom_point()

col_kpi=c('CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'Monthly_Avg_PURCHASES',
        'Monthly_CASH_ADVANCE', 'LIMIT_USAGE', 'MIN_PAYMENTS_RATIO')
cluster_4 <- credit[,col_kpi]
cluster_4 <- cbind(cluster_4,binary_purchase_type)
cluster_4 <- cbind(cluster_4,cluster_df)
cluster_4[1:10,]
dim(cluster_4)

cluster_group_by <- group_by(cluster_4,cluster_df)
cluster_summ <- summarise(cluster_group_by,CASH_ADVANCE_TRX = mean(CASH_ADVANCE_TRX),PURCHASES_TRX= mean(PURCHASES_TRX),
                        Monthly_Avg_PURCHASES = mean(Monthly_Avg_PURCHASES),LIMIT_USAGE = mean(LIMIT_USAGE),Monthly_CASH_ADVANCE = mean(Monthly_CASH_ADVANCE),
                        MIN_PAYMENTS_RATIO = mean(MIN_PAYMENTS_RATIO),purchase_type.both_oneoff_installment = mean(purchase_type.both_oneoff_installment),
                        purchase_type.installment = mean(purchase_type.installment),purchase_type.none = mean(purchase_type.none),
                        purchase_type.oneoff = mean(purchase_type.oneoff))
cluster_summ <- t(cluster_summ)
colnames(cluster_summ) <- c("cluster-1","cluster-2","cluster-3","cluster-4")
cluster_summ
cluster_summ <- cluster_summ[-c(1,2,3),]

cluster_summ <- t(cluster_summ)
cluster_summ <- as.data.frame(cluster_summ)
cluster_summ$Monthly_Avg_PURCHASES <- log(cluster_summ$Monthly_Avg_PURCHASES)
cluster_summ$Monthly_CASH_ADVANCE <- log(cluster_summ$Monthly_CASH_ADVANCE)
cluster_summ <- t(cluster_summ)
cluster_summ <- as.data.frame(cluster_summ)

barplot(as.matrix(cluster_summ), main="Cluster", ylab="values", beside=TRUE, 
      col=terrain.colors(5))

cluster_count <- cluster_scatter %>%
                group_by(Cluster) %>%
                summarise(count_value = n())

cluster_percentage <- cluster_count %>%
                      mutate(percentage = (count_value/sum(count_value)*100))

#-------------------------------

pc_cluster_5 <-kmeans(credit_pca_data, 5)
cluster_df_5 <- (pc_cluster_5$cluster)
pc_cluster_5$centers
pc_cluster_5$size

center_5 <-pc_cluster_5$centers
center_5

cluster_5 <- c(1: 5)
center_df_5 <- data.frame(cluster_5, center_5)
# Reshape the data
center_reshape_5 <- gather(center_df_5, features, values, PC1:PC5)
head(center_reshape_5)

cluster_scatter_5 <- cbind(credit_pca_data,pc_cluster_5$cluster)
names(cluster_scatter_5)[6] <- "Cluster"
ggplot(data=cluster_scatter_5, aes(x=PC1, y=PC2, color=factor(Cluster))) + 
  geom_point()

col_kpi=c('CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'Monthly_Avg_PURCHASES',
          'Monthly_CASH_ADVANCE', 'LIMIT_USAGE', 'MIN_PAYMENTS_RATIO')
cluster_5 <- credit[,col_kpi]
cluster_5 <- cbind(cluster_5,binary_purchase_type)
cluster_5 <- cbind(cluster_5,cluster_df_5)
cluster_5[1:10,]
dim(cluster_5)

cluster_group_by_5 <- group_by(cluster_5,cluster_df_5)
cluster_summ <- summarise(cluster_group_by_5,CASH_ADVANCE_TRX = mean(CASH_ADVANCE_TRX),PURCHASES_TRX= mean(PURCHASES_TRX),
                          Monthly_Avg_PURCHASES = mean(Monthly_Avg_PURCHASES),LIMIT_USAGE = mean(LIMIT_USAGE),Monthly_CASH_ADVANCE = mean(Monthly_CASH_ADVANCE),
                          MIN_PAYMENTS_RATIO = mean(MIN_PAYMENTS_RATIO),purchase_type.both_oneoff_installment = mean(purchase_type.both_oneoff_installment),
                          purchase_type.installment = mean(purchase_type.installment),purchase_type.none = mean(purchase_type.none),
                          purchase_type.oneoff = mean(purchase_type.oneoff))
cluster_summ <- t(cluster_summ)
colnames(cluster_summ) <- c("cluster-1","cluster-2","cluster-3","cluster-4","cluster-5")
cluster_summ
cluster_summ <- cluster_summ[-c(1,2,3),]

cluster_summ <- t(cluster_summ)
cluster_summ <- as.data.frame(cluster_summ)
cluster_summ$Monthly_Avg_PURCHASES <- log(cluster_summ$Monthly_Avg_PURCHASES)
cluster_summ$Monthly_CASH_ADVANCE <- log(cluster_summ$Monthly_CASH_ADVANCE)
cluster_summ <- t(cluster_summ)
cluster_summ <- as.data.frame(cluster_summ)

barplot(as.matrix(cluster_summ), main="Cluster", ylab="values", beside=TRUE, 
        col=terrain.colors(5))
cluster_count <- cluster_scatter %>%
  group_by(Cluster) %>%
  summarise(count_value = n())

cluster_percentage <- cluster_count %>%
  mutate(percentage = (count_value/sum(count_value)*100))

cluster_count
cluster_percentage