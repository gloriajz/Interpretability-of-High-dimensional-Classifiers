# Import necessary libraries
library(ranger)
library(caret)
library(ggplot2)
library(GGally)
library(xgboost)
library(shapr)
library(fastshap)
library(iml)
library(tidyverse)
library(neuralnet)
library(numDeriv)
library(splines)
library(np)
library(gridExtra)
library(NeuralNetTools)

fit.classifiers <- function(train, test) {
  # Logistic Regression
  logistic_model <- glm(as.factor(y) ~ ., data = train, family = binomial(link = "logit"))
  pred_logistic <- predict(logistic_model, test, type = "response")
  
  # Random Forest
  y.colID = which(colnames(train) == "y")
  rf_model <- ranger(x = train[,-y.colID], y = as.factor(train$y), 
                     importance = "impurity", probability = TRUE)
  pred_rf <- predict(rf_model, data = test, type = "response")$predictions[,2]
  
  # Neural Network
  #train.nn.matrix = model.matrix(~., data = train)
  #train.nn.matrix = train.nn.matrix[,-1]
  neural_model <- neuralnet(
    as.factor(y)~.,
    data=train,
    hidden=c(4),
    linear.output = FALSE
  )
  #test.nn.matrix = model.matrix(~., data = test)
  #test.nn.matrix = test.nn.matrix[,-1]
  pred_nn <- predict(neural_model, test, type = "response")[,2]
  
  return(list(logistic = pred_logistic, 
              randomForest = pred_rf, 
              neuralNetwork = pred_nn,
              logistic_model = logistic_model,
              rf_model = rf_model,
              neural_model = neural_model))
}

misclassificationerror <- function(pred, true, threshold){
  return(1-mean((pred > threshold)==true))
}


# Variable Importance plots

# Write a function

variable.importance.plots <- function(fit){
  
  # Logistic Regression Model
  logistic.imp = (summary(fit$logistic_model)$coefficients[,3])
  logistic_importance = data.frame(Variable = names(logistic.imp),
                                   importance = logistic.imp)
  
    
  # Random Forest Model
  importance <- importance(fit$rf_model)
  rf_importance = data.frame(Variable = names(importance),
                              importance = importance)
  
  # NN Model
  nn_imp = olden(fit$neural_model, bar_plot = FALSE)
  nn_importance = data.frame(Variable = rownames(nn_imp),
                             Importance = nn_imp)
  
  return(list(logistic_importance = logistic_importance, 
              rf_importance = rf_importance, 
              nn_importance = nn_importance))
}


# NN plot
nn.plot <- function(fit){
  plot(fit$neural_model,rep = "best")
}


# Write a function for Shapley Value Plots

# First define predictor functions:
# Logistic Regression Model
pfun_glm <- function(glm.fit, newdata){
  
  return(predict(glm.fit, newdata, type = "response"))
}

# Random Forest Model
pfun_ranger <- function(ranger.fit, newdata){
  return(predict(ranger.fit, newdata)$predictions[,2])
}

# Neural Network Model
pfun_neural <- function(neuralnet.fit, newdata){
  predictions <- predict(neuralnet.fit, newdata)
  if(is.matrix(predictions)){
    return(predictions[, 2])
  } else {
    stop("Unexpected output format from predict function")
  }
}

Shapley_value.plots <- function(model, test, pred_wrapper, ...){
  
  explanation <- fastshap::explain(
    model,
    X = test,
    pred_wrapper = pred_wrapper,
    ...
  )
  # Aggregate Shapeley values
  Shapley = apply(explanation, MARGIN = 2, FUN = function(x) sum(abs(x)))
  shap_aggre = data.frame(Variable = names(Shapley),
                          Shapley = Shapley)
  
  return(shap_aggre)
}



# Write function for Active Subspace Plots

active_subspace.plot <- function(data, pred_probs) {
  
  #if(nrow(data) > 500){
    
    #idx = sample(1:nrow(data), 500)
    #smooth = npregbw(xdat = data[idx,], ydat = pred_probs[idx])
    
  #}else{
    
    smooth = npregbw(xdat = data, ydat = pred_probs)
    
  #}
  
  # Compute gradients
  smooth.fit = npreg(smooth, gradients = T, xdat = data, ydat = pred_probs)
  GradMatrix <- smooth.fit$grad

  # Perform mean gradients
  MeanGrad <- rowMeans(GradMatrix)
  mean_grad_df <- data.frame(MeanGrad = MeanGrad, Variable = colnames(data))
  
  
  # Perform PCA on the gradients
  pca <- prcomp(GradMatrix, scale = TRUE)
  
  # Perform eigenvalues
  eigenvectors_df <- data.frame(pca$rotation)
  eigenvectors_df$Variable <- colnames(data)
  
  return(list(mean_grad_df = mean_grad_df, 
              eigenvectors_df = eigenvectors_df,
              pca = pca))
}

