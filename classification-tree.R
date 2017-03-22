library(knitr)
knitr::opts_chunk$set(echo = TRUE)
library(plyr)
library(dplyr)
library(vcd)
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
plot(t$finalModel)
text(t$finalModel)
summary(t$finalModel)

# make predictions
x_test <- titanicTest[,2:9]
y_test <- titanicTest[,1]
predictions <- predict(t, x_test)
# summarize results
confusionMatrix(predictions, y_test)
