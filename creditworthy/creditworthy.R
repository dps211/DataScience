#library(ggplot2)
library(data.table)
#trn_data <- fread("D:/R/creditworthy/sample_data.csv", na.strings = c("","NA"))
trn_data <- read.csv("D:/R/creditworthy/sample_data.csv", na.strings = c("","NA"))
tst_data <- read.csv("D:/R/creditworthy/test_data.csv", na.strings = c("","NA"))
output_data <- read.csv("D:/R/creditworthy/sample_output.csv", na.strings = c("","NA"))
merge_trn_data <- merge(x=trn_data, y=output_data, by.x="Application_ID", by.y="Application_ID")
View(merge_trn_data)
df_tst <- data.frame(tst_data)

#data$Age[is.na(data$Age)] <- mean(data$Age,na.rm=T)
#trn_data_cleaned <- merge_trn_data[complete.cases(merge_trn_data),] #removing the row which has blanks and NA
#dim(trn_data_cleaned)# 80,12 20 rows removed
#View(trn_data_cleaned)
#Recoding categorical data
df <- data.frame(merge_trn_data)
str(df)
summary(df)
View(df)

df$Gender[is.na(df$Gender)] <- "M"
df$Self_Employed[is.na(df$Self_Employed)] <- "No"
df$LoanAmount[is.na(df$LoanAmount)] <- mean(df$LoanAmount,na.rm=T)
df$Loan_Amount_Term[is.na(df$Loan_Amount_Term)] <- 360
df$Credit_History[is.na(df$Credit_History)] <- 1
View(df)
df$Gender <- ifelse(df$Gender == "M", 1, 2)
df$Married <- ifelse(df$Married == "Yes", 1, 2)
df$Dependents <- ifelse(df$Dependents == "3+", 3, df$Dependents)

df$Education <- ifelse(df$Education == "Graduate", 1, 2)
df$Self_Employed <- ifelse(df$Self_Employed == "Yes", 1, 2)
df$Property_Area <- ifelse(df$Property_Area == "Urban", 1, ifelse(df$Property_Area == "Semiurban", 2, 3))
View(df)
dim(df)
#imputing test data
df_tst$Gender[is.na(df_tst$Gender)] <- "M"
df_tst$Self_Employed[is.na(df_tst$Self_Employed)] <- "No"
df_tst$Married[is.na(df_tst$Married)] <- "Yes"
df_tst$Dependents[is.na(df_tst$Dependents)] <- 0
df_tst$LoanAmount[is.na(df_tst$LoanAmount)] <- mean(df_tst$LoanAmount,na.rm=T)
df_tst$Loan_Amount_Term[is.na(df_tst$Loan_Amount_Term)] <- 360
df_tst$Credit_History[is.na(df_tst$Credit_History)] <- 1
#recoding test data. Could be done in a common function for train and test
df_tst$Gender <- ifelse(df_tst$Gender == "M", 1, 2)
df_tst$Married <- ifelse(df_tst$Married == "Yes", 1, 2)
df_tst$Dependents <- ifelse(df_tst$Dependents == "3+", 3, df_tst$Dependents)

df_tst$Education <- ifelse(df_tst$Education == "Graduate", 1, 2)
df_tst$Self_Employed <- ifelse(df_tst$Self_Employed == "Yes", 1, 2)
df_tst$Property_Area <- ifelse(df_tst$Property_Area == "Urban", 1, ifelse(df_tst$Property_Area == "Semiurban", 2, 3))


prediction_1 <- glm(Loan_Status ~ Gender + Married + Dependents + 
                      Education + Self_Employed + ApplicantIncome + 
                      CoapplicantIncome + LoanAmount + Loan_Amount_Term + Credit_History + 
                      Property_Area, family = binomial(link = "logit"), data=df)
summary(prediction_1)

#out of all the Married (Marital Status) and Property_Area has lower p value rest other parameters are not
# statistically significant

#ln(odds) = ln(p/1-p) = -1.51 *Married - -1.14 * Property_Area
#anova(prediction_1, test="Chisq")

fitted_results <- predict(prediction_1,newdata = df_tst, type = 'response' )

fitted_results 
fitted_results_1 <- ifelse(fitted_results > 0.5, "Y", "N")
fitted_results_1

final_output <- cbind(df_tst[,1],fitted_results_1)
df_final_output <- data.frame(final_output)

names(df_final_output)[1] <- paste("Application_ID")
names(df_final_output)[2] <- paste("Loan_Status")
df_final_output

write.csv(df_final_output, file = "D:/R/creditworthy/test_output.csv")



