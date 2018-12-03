#Date: 20/02/2017
#Author: Dharmendra Singh
#Version: 1.0
#Problem statement - Build a model using to predict housing prices
######################################################################
#load important packages
library(ggplot2)
library(data.table)

#reading data
#trn_data <- fread("D:/R/certace/sample_data.csv", na.strings = c("","NA"))
all_data <- read.csv("D:/R/certace/all.csv") # 100,000 obs of 13 vars
summary(all_data)
str(all_data)
View(all_data)
df_all_data <- data.frame(all_data, stringsAsFactors = FALSE)
df_all_data$Property_Type <- ifelse(df_all_data$Property_Type == "D", 1, ifelse(df_all_data$Property_Type == "S", 2, ifelse(df_all_data$Property_Type == "T", 3, ifelse(df_all_data$Property_Type == "F", 4, 5))))
df_all_data$Old_New <- ifelse(df_all_data$Old_New == "Y", 1, 2)
df_all_data$Duration <- ifelse(df_all_data$Duration == "F", 1, ifelse(df_all_data$Property_Type == "L", 2, 3))
df_all_data$PPD_Category_Type <- ifelse(df_all_data$PPD_Category_Type == "A", 1, 2)
View(df_all_data)


str(df_all_data)
train_data <- df_all_data[1:98818,] # first 98818 records are for train
dim(train_data)
tail(train_data)
write.csv(train_data, file = "D:/R/certace/train_data.csv")
#Outlier Detection
train_data_outlier_detection <- df_all_data[1:98818,] # first 98818 records are for train
source("http://goo.gl/UUyEzD")
outlierKD(train_data_outlier_detection, Price)
summary(train_data_outlier_detection)
dim(train_data_outlier_detection)

# replacing NA with mean calculated without outliers
train_data_outlier_detection$Price[is.na(train_data_outlier_detection$Price)] <- mean(train_data_outlier_detection$Price, na.rm=T)
summary(train_data_outlier_detection)
train_data_with_impu_mean <- train_data_outlier_detection
summary(train_data_with_impu_mean)
dim(train_data_with_impu_mean)
ggplot(data=train_data_with_impu_mean) + geom_histogram(aes(x=Price), binwidth = 10000)

tst_data <- df_all_data[98819:100000,] # remaining 1182 records are for test
write.csv(tst_data, file = "D:/R/certace/tst_data.csv")
dim(tst_data)
head(tst_data)

missing_value_check <- train_data[is.na(train_data),]
dim(missing_value_check) # dim 0 13, means no missing value
summary(train_data)


gc()
#prediction_1 <- glm(Price ~ Gender + Married + Dependents +                       Education + Self_Employed + ApplicantIncome +                       CoapplicantIncome + LoanAmount + Loan_Amount_Term + Credit_History +                       Property_Area, family = binomial(link = "logit"), data=df)

#prediction_1 <- glm(Price ~ Date + Postcode + Property_Type + Old_New 
#                    + Duration + Street + Locality + Town + District + County + PPD_Category_Type,
#                    family = binomial(link = "logit"), data = train_data_with_impu_mean)
#prediction_temp <- lm(formula = Price ~ Date + Postcode + Property_Type + Old_New 
#                    + Duration + Street + Locality + Town + District + County + PPD_Category_Type,
#                     data = train_data_with_impu_mean)
#set.seed(21175)
#samp <- sample(train_data_with_impu_mean, 5)
#dim(samp)
subset1 <- train_data_with_impu_mean[,]
dim(subset1)
dim(train_data_with_impu_mean)
#prediction_2 <- glm(Price ~ Property_Type + Old_New + Duration + PPD_Category_Type, 
 #                   family = binomial(link = "logit"), data = subset1) 

prediction_temp <- lm(formula = Price ~ Date + Postcode + Property_Type + Old_New 
                    + Duration + Street + Locality + Town + District + County + PPD_Category_Type,
                     data = subset1)
prediction_temp

#predict <- lm(formula = Price ~ Property_Type + Old_New + Duration + PPD_Category_Type, data = subset1)
#predict <- lm(formula = Price ~ Property_Type + Old_New + Duration,  data = subset1)
predict_price <- lm(formula = Price ~ Property_Type + Old_New + Duration,  data = train_data_with_impu_mean)
predict_price
summary(predict_price)
View(tst_data)
results <- predict(predict_price,newdata = tst_data, type = 'response' )
results_11 <- predict.lm(predict_price,newdata = tst_data)
results
results_11
View(results)
df_result <- data.frame(results_11)
dim(df_result)

df_result
View(df_result)
final_result <- cbind(tst_data[,1],df_result)
final_result
names(final_result)[1] <- paste("ID")
names(final_result)[2] <- paste("Price")

write.csv(final_result, file = "D:/R/certace/final_output.csv")

#out of all the Married (Marital Status) and Property_Area has lower p value rest other parameters are not
# statistically significant

#ln(odds) = ln(p/1-p) = -1.51 *Married - -1.14 * Property_Area
#anova(prediction_1, test="Chisq")

fitted_results <- predict(prediction_1,newdata = df_tst, type = 'response' )
fitted_results_new1 <- predict(predict_price,newdata = df_tst)

fitted_results 
fitted_results_new1
fitted_results_1 <- ifelse(fitted_results > 0.5, "Y", "N")
fitted_results_1

final_output <- cbind(df_tst[,1],fitted_results_1)
df_final_output <- data.frame(final_output)

names(df_final_output)[1] <- paste("Application_ID")
names(df_final_output)[2] <- paste("Loan_Status")
df_final_output

#Writing final output
write.csv(df_final_output, file = "D:/R/creditworthy/test_output.csv")



