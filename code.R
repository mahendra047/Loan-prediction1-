
##objective:- Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form.

(train <- read.csv("train_loan.csv",header =TRUE,sep=",",na.strings=c(""," ",NA)))
#na.strings: a character vector of strings which are to be interpreted as NA values
View(train)
(test<- read.csv("test_loan.csv",header=TRUE,sep=",",na.strings=c(""," ",NA)))
install.packages("mlr")
library(mlr)
summarizeColumns(train) ##alternate to summary function
summarizeColumns(test)
#loan status is the target variable
#### take out target variable because it should not be standardized or dummified
Loan_Status <- data.frame(train$Loan_Status)
#### remove the variables which are not important for prediction eg: id
train <- train[,-1]
View(train)
testid <- test[,1]
test <- test[,-1]
###MISSING VALUE ANALYSIS
install.packages("Amelia")
library(Amelia)
missmap(train,c("RED","BLUE"))
str(train)
##checking missing value data column wise
misval_train <- data.frame(sapply(train, function(x) sum(is.na(x))))
misval_test <- data.frame(sapply(test, function(x) sum(is.na(x))))
#### as mode function is not pre-defined, we define our own. this will be useful for imputing categorical data
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
#### impute the categorical variables by taking mode for that column
train$Gender[is.na(train$Gender)] <- getmode(train$Gender)
train$Married[is.na(train$Married)] <- getmode(train$Married)
train$Dependents[is.na(train$Dependents)] <- getmode(train$Dependents)
train$Self_Employed[is.na(train$Self_Employed)] <- getmode(train$Self_Employed)
train$Loan_Amount_Term[is.na(train$Loan_Amount_Term)] <- getmode(train$Loan_Amount_Term)
train$Credit_History[is.na(train$Credit_History)] <- getmode(train$Credit_History)
#### impute the numerical variables by taking mean for that column
train$LoanAmount[is.na(train$LoanAmount)] <- round(mean(train$LoanAmount, na.rm = TRUE))
#### imputing test data
test$Gender[is.na(test$Gender)] <- getmode(test$Gender)
test$Dependents[is.na(test$Dependents)] <- getmode(test$Dependents)
test$Self_Employed[is.na(test$Self_Employed)] <- getmode(test$Self_Employed)
test$Loan_Amount_Term[is.na(test$Loan_Amount_Term)] <- getmode(test$Loan_Amount_Term)
test$Credit_History[is.na(test$Credit_History)] <- getmode(test$Credit_History)

test$LoanAmount[is.na(test$LoanAmount)] <- round(mean(test$LoanAmount, na.rm = TRUE))
rm(misval_train,misval_test)
#### converting loan_amt_term and credit_history to categorical
train$Credit_History <- as.factor(train$Credit_History)

####
test$Credit_History <- as.factor(test$Credit_History)
####
#creating new variable :- Family_income
train$Family_Income <- train$ApplicantIncome + train$CoapplicantIncome
train$ApplicantIncome <- NULL
train$CoapplicantIncome <- NULL
####
test$Family_Income <- test$ApplicantIncome + test$CoapplicantIncome
test$ApplicantIncome <- NULL
test$CoapplicantIncome <- NULL
#### seperate the data into numeric and categorical variables
train_num <- train[sapply(train,is.numeric)]
train_cat <- train[sapply(train,is.factor)]
names(train_cat)
View(train)
head(train)
View(train_cat)
# column 8 is our target, we dont want to dummify our target variable, hence remove it from train_cat
train_cat <- subset(train_cat, select = -8)
head(train_cat,2)
install.packages(c("dummies","vegan"))
library(dummies)
library(vegan)
train_num_std <- decostand(train_num, method = "standardize")
train_cat_dummy <- dummy.data.frame(train_cat)
####
final_train <- cbind(train_num_std,train_cat,Loan_Status)
colnames(final_train)[11] <- "Loan_Status"
final_train_sd <- cbind(train_num_std,train_cat_dummy,Loan_Status)
colnames(final_train_sd)[21] <- "Loan_Status"
####### plot to see the corelations
library(ggplot2)
install.packages("GGally")
library(GGally)
##
plot = ggpairs(train_num, mapping = aes(colour = "dark green"),
               axisLabels = "show")
print(plot)
################# dividing test data into numeric and categorical #################
test_num <- test[sapply(test,is.numeric)]
test_cat <- test[sapply(test,is.factor)]

test_num_std <- decostand(test_num, method = "standardize")
test_cat_dummy <- dummy.data.frame(test_cat)

final_test_sd <- cbind(test_num_std,test_cat_dummy)
#### divide into train and validate by taking a stratified sample
library(caret)
datapart <- createDataPartition(final_train$Loan_Status,p=0.7,list = F)
new_train <- final_train[datapart,]
new_validate <- final_train[-datapart,]

new_train_pr <- final_train_sd[datapart,]
new_validate_pr <- final_train_sd[-datapart,]
#k-fold cross validation
train_control <- trainControl(method = "repeatedcv", number = 10,repeats = 10)
model <- train(Loan_Status ~ ., data=final_train, trControl=train_control, method="rf", metric="Accuracy")
print(model)
####
x <- subset(new_train, select = -11)
y <- as.factor(new_train[,11])
#### build model using rpart algorithm
library(rpart)
model_rpart <- rpart(Loan_Status ~ ., data = new_train, method = "class",cp=0.01)
summary(model_rpart)
#### predict on new_validate
pred_rpart <- predict(model_rpart,new_validate, type = "class")
acc_rpart <- confusionMatrix(new_validate$Loan_Status,pred_rpart)$overall[1]
### build using knn
library(class)
pred_knn <- knn(new_train_pr[1:20],new_validate_pr[1:20],new_train_pr$Loan_Status,k = 7)
acc_knn <- confusionMatrix(new_validate_pr$Loan_Status,pred_knn)$overall[1]
#### build using svm
x_sd <- subset(new_train_pr, select = -21)
y_sd <- as.factor(new_train_pr$Loan_Status)

a_sd <- subset(new_validate_pr, select = -21)
b_sd <- as.factor(new_validate_pr$Loan_Status)
library(e1071)
model_svm <- svm(x_sd, y_sd, kernel = "linear") 
pred_svm <- predict(model_svm,a_sd)

acc_svm <- confusionMatrix(b_sd,pred_svm)$overall[1]
##### tuning svm to get best results
tuneResult <- tune(svm, train.x = x_sd, train.y = y_sd, ranges = list(gamma = 10^(-6:-1), cost = 2^(2:3)))
print(tuneResult)
tunedModel <- tuneResult$best.model 
tunedModelY <- predict(tunedModel, as.matrix(x_sd)) 
Conf <- table(y_sd, tunedModelY)
confusionMatrix(y_sd,tunedModelY)$overall[1]
########################## predicting on svm
pred_svm_test <- predict(model_svm,as.matrix(final_test_sd))
final_result_svm <- data.frame(pred_svm_test)
final_result_svm <- cbind(testid,final_result_svm)
colnames(final_result_svm)[1] <- "Loan_ID"
colnames(final_result_svm)[2] <- "Loan_Status"
write.csv(final_result_svm,"FinalResult1.csv")
#### bind accuracy of all models to compare
accuracy <- cbind(acc_svm,acc_knn,acc_rpart)



