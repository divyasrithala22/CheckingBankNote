if (!requireNamespace("tidyverse")) install.packages('tidyverse')
if (!requireNamespace("caret")) install.packages('caret')
if (!requireNamespace("neuralnet")) install.packages('neuralnet')
if (!requireNamespace("keras")) install.packages('keras')
if (!requireNamespace("randomForest")) install.packages('randomForest')
if (!requireNamespace("rpart")) install.packages('rpart')
if (!requireNamespace("rattle")) install.packages('rattle')

library(tidyverse)
library(caret)
library(neuralnet)
library(keras)
library(randomForest)
library(rpart)
library(rattle)

getwd()
wd4 <- "C:/Users/divya/Desktop/CheckingBankNote"
setwd(wd4)
banknote <- read.csv('banknote.csv',header=TRUE)
View(banknote)
banknote <- na.omit(banknote)

set.seed(123)
training.samples <- banknote$class %>% createDataPartition(p = 0.75, list = FALSE)
train.data  <- banknote[training.samples, ]
test.data <- banknote[-training.samples, ]

set.seed(123)
model <- neuralnet(class~., data = train.data, hidden = 0, err.fct = "sse", linear.output = F)
plot(model, rep = "best")

probabilities <- model %>% predict(test.data) %>% as.vector()
predicted.classes <- ifelse(probabilities > 0.5, 1, 0)
confusionMatrix(factor(predicted.classes), factor(test.data$class), positive = '1')

set.seed(123)
model <- neuralnet(class~., data = train.data, hidden = 0, err.fct = "ce", linear.output = F)
plot(model, rep = "best")

probabilities <- model %>% predict(test.data)
predicted.classes <- ifelse(probabilities > 0.5, 1, 0)
confusionMatrix(factor(predicted.classes), factor(test.data$class), positive = '1')

set.seed(123)
model <- glm(class~., family = binomial, data = train.data)
model

###The CE Loss function model better ensembles logistic regression model.

probabilities <- model %>% predict(test.data, type = 'response')
predicted.classes <- ifelse(probabilities > 0.5, 1, 0)
confusionMatrix(factor(predicted.classes), factor(test.data$class), positive = '1')

set.seed(123)
model <- neuralnet(class~., data = train.data, hidden = 3, err.fct = "sse", linear.output = F)
plot(model, rep = "best")

probabilities <- model %>% predict(test.data)
predicted.classes <- ifelse(probabilities > 0.5, 1, 0)
confusionMatrix(factor(predicted.classes), factor(test.data$class), positive = '1')

###The prediction with hidden layer is better than no hidden layer.`

set.seed(123)
model <- neuralnet(class~., data = train.data, hidden = 3, err.fct = "ce", linear.output = F)
plot(model, rep = "best")

probabilities <- model %>% predict(test.data)
predicted.classes <- ifelse(probabilities > 0.5, 1, 0)
confusionMatrix(factor(predicted.classes), factor(test.data$class), positive = '1')

###The prediction with hidden layer is better than no hidden layer.

########### To conduct the random forest, need factorize the response data, or will become regression random forest###########
train.data$class <- factor(train.data$class)
test.data$class <- factor(test.data$class)
##################################

set.seed(123)
model <- train(
  class ~., data = train.data, method = "rf",
  trControl = trainControl("cv", number = 10),
  importance = TRUE
)
### Best tuning parameter

model$bestTune
model$finalModel

pred <- model %>% predict(test.data)
#predict(model, test)
confusionMatrix(pred, test.data$class, positive = '1')

### Plot MeanDecreaseAccuracy
varImpPlot(model$finalModel, type = 1)
### Plot MeanDecreaseGini
varImpPlot(model$finalModel, type = 2)


varImp(model, type = 1)

###It should be the square root of the total variables
sqrt(25)
model <- rpart(class ~., data = train.data, control = rpart.control(cp=0))
par(xpd = NA)
fancyRpartPlot(model)
pred_full <- predict(model,newdata = test.data, type ='class')
confusionMatrix(pred_full, test.data$class)
set.seed(123)
model2 <- train(
  class ~., data = train.data, method = "rpart",
  trControl = trainControl("cv", number = 10),
  tuneLength = 100)
plot(model2)

model2$bestTune

###This dataset is a little bit special, the best pruned tree is just the fully grown tree.


fancyRpartPlot(model2$finalModel)
pred_prune <- predict(model2, newdata = test.data)
confusionMatrix(pred_prune, test.data$class)

### if the pruned tree is different with fully grown tree, it should not be 1
mean(pred_full == pred_prune)