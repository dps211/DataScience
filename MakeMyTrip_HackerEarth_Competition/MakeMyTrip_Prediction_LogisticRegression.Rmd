---
title: "R Notebook"
output: html_notebook
---

This is an [R Markdown](http://rmarkdown.rstudio.com) Notebook. When you execute code within the notebook, the results appear beneath the code. 

Try executing this chunk by clicking the *Run* button within the chunk or by placing your cursor inside it and pressing *Ctrl+Shift+Enter*. 

```{r}
plot(cars)
```

Add a new chunk by clicking the *Insert Chunk* button on the toolbar or by pressing *Ctrl+Alt+I*.

When you save the notebook, an HTML file containing the code and output will be saved alongside it (click the *Preview* button or press *Ctrl+Shift+K* to preview the HTML file).

The preview shows you a rendered HTML copy of the contents of the editor. Consequently, unlike *Knit*, *Preview* does not run any R code chunks. Instead, the output of the chunk when it was last run in the editor is displayed.
#-------------------
#Predict value of P
#-------------------

#Data visualization
library(ggplot2)
library(data.table)

#Data wrangling
library(mice)
library(dummies)
library(caret)
library(caTools)

#Model building, Evaluation
library(rpart)
library(rpart.plot)
library(MLmetrics)

#Set working directory
setwd("D:/R/MMT/datasete9dc3ed/dataset/")

## Reading the file
mmt_trn_data <- read.csv("train.csv", na.strings = c("","NA"))
```{r}
dim(mmt_trn_data)

```
```{r}
head(mmt_trn_data)
```

```{r}
str(mmt_trn_data)
```

```{r}
summary(mmt_trn_data)
```
```{r}
md.pattern(mmt_trn_data)
```
#mmt_trn_data_na_rem <- na.omit(mmt_trn_data)
```{r}
#dim(mmt_trn_data_na_rem)
```
```{r}
#Data imputation
imputed_mmt_trn_data <- mice(mmt_trn_data, print = F, seed = 123)
imp_mmt_trn_data <- complete(imputed_mmt_trn_data)
md.pattern(imp_mmt_trn_data)
```
```{r}
#EXPLORATORY ANALYSIS
g <- ggplot(imp_mmt_trn_data, aes(x=B,y=O))
g + geom_point(aes(color = E)) + facet_wrap(~E)


```
#gg in E column seems outlier, should be removed
```{r}
g + geom_point(aes(color = E)) + facet_grid(D~G)
```

#n - 1and o 2 in G column seems outlier
```{r}
g + geom_point(aes(color = E)) + facet_wrap(~F)
##r in F column seems outlier
```
```{r}
densityplot(imp_mmt_trn_data$B) 
densityplot(log(imp_mmt_trn_data$B))

```
```{r}
#
densityplot(imp_mmt_trn_data$C) 
densityplot(log(imp_mmt_trn_data$C))

```
```{r}
densityplot(imp_mmt_trn_data$H) 
densityplot(log(imp_mmt_trn_data$H))
```
```{r}
densityplot(imp_mmt_trn_data$K) 
densityplot(log(imp_mmt_trn_data$K))
```
```{r}
densityplot(imp_mmt_trn_data$N) 
densityplot(log(imp_mmt_trn_data$N))
```
```{r}
densityplot(imp_mmt_trn_data$O) 
densityplot(log(imp_mmt_trn_data$O)) 
#unselecting outlier id = 129
imp_mmt_trn_data_rem_olr <- imp_mmt_trn_data[imp_mmt_trn_data$id != 129,]
densityplot(imp_mmt_trn_data_rem_olr$O)
densityplot(log(imp_mmt_trn_data_rem_olr$O))
#removing outlier doesn't change the distribution so retaining id = 129
```
# Variable transformation of nominal variable
nom_feature <- c("B","C","H","K","N","O")
temp_imp_mmt_trn_data <- imp_mmt_trn_data
#replaing zeros of nominal column with smaller value which will be be closer to zero 
#after log transformation as zeros of log transformation would result in infinity
temp_imp_mmt_trn_data[temp_imp_mmt_trn_data == 0] <- 1.001
temp_imp_mmt_trn_data[nom_feature] <- log(temp_imp_mmt_trn_data[nom_feature])
```{r}
head(temp_imp_mmt_trn_data)
```


#Adding dummy variables for categorical data
imp_mmt_trn_data_dummy_var <- model.matrix(~ A  + D + E + F + G + J + I + J + L + M + 0, data=imp_mmt_trn_data, 
                      contrasts.arg = lapply(data.frame(imp_mmt_trn_data[,sapply(data.frame(imp_mmt_trn_data),is.factor)]), 
                                             contrasts, contrasts = FALSE))
merge_imp_mmt_trn_data <-  cbind(temp_imp_mmt_trn_data,imp_mmt_trn_data_dummy_var)

final_merge_imp_mmt_trn_data <- subset(merge_imp_mmt_trn_data, select = -c(A, D , E , F , G , J , I , J , L , M))
```{r}
head(imp_mmt_trn_data_dummy_var)
```
```{r}
head(merge_imp_mmt_trn_data)
```
```{r}
head(final_merge_imp_mmt_trn_data)
```

#This data doesn't have original categorical columns and nominal data has been transformed too