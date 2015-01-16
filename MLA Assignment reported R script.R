library(caret)
#library(rpart)
library(randomForest)
setwd("~/Training/Machine learning")

set.seed(673)

#Load data
url<- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(url, destfile = "./trainRaw.csv", method = "curl")
Raw<- read.csv("trainRaw.csv")

inTrain<- createDataPartition(y=Raw$classe, p=0.9, list=FALSE)
trainRaw<- Raw[inTrain,]
valRaw<- Raw[-inTrain,]

#Data basic explorations for formatting
#Kick out X, user_name, timestamps, near zero variables (too many NAs, blanks, #Div/0, - mostly kurtosis / skewness / amplitude)
trainy<- trainRaw$classe

nsv<-  nearZeroVar(trainRaw, saveMetrics=TRUE)
include<- which(nsv$nzv==FALSE) #use nzv column as source of truth

training<- trainRaw[,include]

exclude<- c(1:6) #generic data uncorrelated to output
NALimit<- dim(training)[1]*0.4 #set limit on when to ignore a column if too many NAs : 40%
for  (i in 1:c(dim(training)[2])) {
  s<- (sum(is.na(training[,i])) > NALimit)*1
  if (s==1) {exclude<- c(exclude,i)} else{exclude<-exclude}
}

exclude<- c(exclude, c(dim(training)[2])) #remove output varianvel from predictors data frame
training<- training[,-exclude] #52 variables left

trainPP<- training

modRF<- randomForest(trainy~., data=trainPP, ntree=20, norm.votes=FALSE)
confusionMatrix(predict(modRF, trainPP), trainy)

########################
###Validate

valing<- valRaw[,include]
valy<- valing$classe
valing<- valing[,-exclude] #52 variables left
valPP<- valing

confusionMatrix(predict(modRF, valPP), valy)

######################
#######################
# Download and process testing dataset
url<- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url, destfile = "./testRaw.csv", method = "curl")
testRaw<- read.csv("testRaw.csv")
testing<- testRaw[,include]
testing<- testing[,-exclude] #52 variables left
testPP<- testing

predict(modRF, testPP)
