library(knitr)
knitr::opts_chunk$set(echo = TRUE)
library(plyr)
library(dplyr)
library(AppliedPredictiveModeling)
library(caret)
library(ellipse)
library(kknn)
library(gridExtra)
library(grid)
library(randomForest)
set.seed(3456)
.pardefault <- par()

data(iris)
formula <- as.formula(Species ~.)
t <- train(formula,iris,method = "rpart",cp=0.002,maxdepth=8)
plot(t$finalModel)
text(t$finalModel)
summary(t$finalModel)

# With titanic data
setwd("C:/Users/thimo/git/rwml-R/R")
titanic <- read.csv("../data/titanic.csv", 
                    colClasses = c(
                      Survived = "factor",
                      Name = "character",
                      Ticket = "character",
                      Cabin = "character"))
titanic$Survived <- revalue(titanic$Survived, c("0"="no", "1"="yes"))

titanicTidy <- subset(titanic, select = -c(PassengerId, Name, Ticket, Cabin))

titanicTidy$Age[is.na(titanicTidy$Age)] <- -1

titanicTidy <- titanicTidy %>%
  mutate(sqrtFare = sqrt(Fare)) %>%
  select(-Fare)

titanicTidy <- titanicTidy %>%
  filter(!(Embarked=="")) %>%
  droplevels

dummies <- dummyVars(" ~ .", data = titanicTidy, fullRank = TRUE)
titanicTidyNumeric <- data.frame(predict(dummies, newdata = titanicTidy))

titanicTidyNumeric$Survived.yes <- factor(titanicTidyNumeric$Survived.yes)

# split
trainIndex <- createDataPartition(titanicTidyNumeric$Survived.yes, p = .8, 
                                  list = FALSE, 
                                  times = 1)

titanicTrain <- titanicTidyNumeric[ trainIndex,]
titanicTest  <- titanicTidyNumeric[-trainIndex,]

formula <- as.formula(Survived.yes ~.)
t <- train(formula,titanicTrain,method = "rpart",cp=0.002,maxdepth=8)
trf <- train(formula,titanicTrain,method = "rf")
plot(t$finalModel)
text(t$finalModel)
summary(t$finalModel)

# Plot with partykit and using specific classifier method
rpart1 <-rpart(formula,titanicTrain)
library(partykit)
tparty <- as.party(rpart1)
plot(tparty)


# make predictions
x_test <- titanicTest[,2:9]
y_test <- titanicTest[,1]
predictions <- predict(t, x_test)
predictionsrf <- predict(trf, x_test)
# summarize results
confusionMatrix(predictions, y_test)
varImp(trf)

# Predict with probabilities
predictions_probs <- predict(t, x_test,type="prob")
head(predictions_probs)

# Plot ROC curve
library(pROC)
rpartROC <- roc(y_test, predictions_probs[, "0"], levels = c("0", "1"))
plot(rpartROC, type = "S", print.thres = "best")
auc(rpartROC)

trandomForest <- randomForest(formula, titanicTrain)
varImpPlot(trandomForest)
trandomForestProb <- predict(trandomForest, x_test, type="prob")
rfROC <- roc(y_test, trandomForestProb[, "0"], levels = c("0", "1"))
plot(rfROC, type = "S", print.thres = "best")
auc(rfROC)
