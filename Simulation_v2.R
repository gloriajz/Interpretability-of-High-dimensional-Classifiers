rm(list = ls())
source("functions.R")

slope = -1
intercept1 = 0.5
intercept2 = -0.5
N = 200
d = 5
B = 20
fix.sign = 1

plot(V1, V2)

simulation2_plot = plot(data[,1], data[,2], col = data$y, pch = 16)
ggsave("Simulation2_Plot.png",
       plot = simulation2_plot,
       device = "png",
       width = 10,
       height = 4)

glm.error = rep(0, B)
rf.error = rep(0, B)
nn.error = rep(0, B)
logistic_importance = matrix(0, nrow = 0, ncol = 3)
rf_importance = matrix(0, nrow = 0, ncol = 3)
nn_importance = matrix(0, nrow = 0, ncol = 3)
shap_glm = list()
shap_ranger = list()
shap_neural = list()
active.glm = list()
active.rf = list()
active.nn = list()
mean_glm = list()
mean_rf = list()
mean_nn = list()
eigenvalues_glm = list()
eigenvalues_rf = list()
eigenvalues_nn = list()

for(i in 1:B){
  
  # Create two datasets, V1 and V2
  
  V1 = runif(N, 0, 1)
  id = rbinom(N, 1, 0.5)
  V2 = slope*V1 + intercept1 + (intercept2 - intercept1)*id + rnorm(N, 0, 0.1)
  
  sample1 = cbind(cbind(V1, V2), matrix(runif(N*(d-2), 0, 1), ncol = d-2))
  sample1[,2] = 0.5*(sample1[,2] + 1.5)
  sample2 = matrix(runif(N*d, 0, 1), ncol = d)
  
  #sample1_df <- as.data.frame(sample1)
  #sample2_df <- as.data.frame(sample2)
  
  #sample1_df$y <- rep(1, N)
  #sample2_df$y <- rep(0, N)

  
  # Combine sample1 and sample2 into a single dataframe
  
  data = as.data.frame(rbind(sample1, sample2))
  data$y = as.factor(c(rep(0, N), rep(1, N)))
  
  # Use part of data as training and part as testing
  
  samp = sample(nrow(data), floor(nrow(data)/2))
  train = data[samp,]
  test = data[-samp,]
  
  # Fit Classifiers
  
  fit.simulation = fit.classifiers(train, test)
  
  # Misclassification error
  
  #true = as.numeric(test$y) - 1
  true = as.numeric(test$y)
  glm.error[i] = 1 - misclassificationerror(fit.simulation$logistic, true, threshold = 0.5)
  rf.error[i] = 1 - misclassificationerror(fit.simulation$randomForest, true, threshold = 0.5)
  nn.error[i] = 1 - misclassificationerror(fit.simulation$neuralNetwork, true, threshold = 0.5)
  
  # Variable Importance plots
  
  plots = variable.importance.plots(fit.simulation)
  plots$logistic_importance[,2] = sign(plots$logistic_importance[fix.sign+1, 2])*plots$logistic_importance[,2]
  logistic_importance = rbind(logistic_importance, 
                              cbind(rep(i, d+1), plots$logistic_importance))
  plots$rf_importance[,2] = sign(plots$rf_importance[fix.sign, 2])*plots$rf_importance[,2]
  rf_importance = rbind(rf_importance, 
                        cbind(rep(i, d), plots$rf_importance))
  plots$logistic_importance[,2] = sign(plots$logistic_importance[fix.sign+1, 2])*plots$logistic_importance[,2]
  plots$nn_importance[,2] = sign(plots$nn_importance[fix.sign, 2])*plots$nn_importance[,2]
  nn_importance = rbind(nn_importance, 
                        cbind(rep(i, d), plots$nn_importance))
  
  # Shapley Values
  # Logistic Regression Plot
  
  shap_glm[[i]] = Shapley_value.plots(fit.simulation$logistic_model, test = test[,1:d], 
                                      pred_wrapper = pfun_glm, exact = TRUE)
  
  # Random Forest Plot
  
  shap_ranger[[i]] = Shapley_value.plots(fit.simulation$rf_model, test = test[,1:d], 
                                         pred_wrapper = pfun_ranger, nsim = 100)
  
  # Neural Network Plot
  
  shap_neural[[i]] = Shapley_value.plots(fit.simulation$neural_model, test = test[,1:d], 
                                         pred_wrapper = pfun_neural, nsim = 100)
  
  # Active Subspace Plots
  # Mean gradient and Eigenvalues plot
  # Logistic Regression
  active.glm[[i]] = active_subspace.plot(data = test[,1:d], pred_probs = fit.simulation$logistic)
  active.glm[[i]]$mean_grad_df[,1] = sign(active.glm[[1]]$mean_grad_df[fix.sign, 1])*active.glm[[i]]$mean_grad_df[,1]
  mean_glm = rbind(active.glm[[i]]$mean_grad_df)
  
  active.glm[[i]]$eigenvectors_df[,1] = sign(active.glm[[1]]$eigenvectors_df[fix.sign, 1])*active.glm[[i]]$eigenvectors_df[,1]
  eigenvalues_glm = rbind(active.glm[[i]]$eigenvectors_df)
  
  # Random Forest
  active.rf[[i]] = active_subspace.plot(data = test[,1:d], pred_probs = fit.simulation$randomForest)
  active.rf[[i]]$mean_grad_df[,1] = sign(active.rf[[1]]$mean_grad_df[fix.sign, 1])*active.rf[[i]]$mean_grad_df[,1]
  mean_rf = rbind(active.rf[[i]]$mean_grad_df)
  active.rf[[i]]$eigenvectors_df[,1] = sign(active.rf[[1]]$eigenvectors_df[fix.sign, 1])*active.rf[[i]]$eigenvectors_df[,1]
  eigenvalues_rf = rbind(active.rf[[i]]$eigenvectors_df)
  
  # Neural Network
  active.nn[[i]] = active_subspace.plot(data = test[,1:d], pred_probs = fit.simulation$neuralNetwork)
  active.nn[[i]]$mean_grad_df[,1] = sign(active.nn[[1]]$mean_grad_df[fix.sign, 1])*active.nn[[i]]$mean_grad_df[,1]
  mean_nn = rbind(active.nn[[i]]$mean_grad_df)
  active.nn[[i]]$eigenvectors_df[,1] = sign(active.nn[[1]]$eigenvectors_df[fix.sign, 1])*active.nn[[i]]$eigenvectors_df[,1]
  eigenvalues_nn = rbind(active.nn[[i]]$eigenvectors_df)
}

logistic_importance = as.data.frame(logistic_importance)
rf_importance = as.data.frame(rf_importance)
nn_importance = as.data.frame(nn_importance)

p.logistic = ggplot(logistic_importance, aes(x = Variable, y = importance)) + 
  geom_violin()+
  geom_boxplot(width = 0.1) +
  geom_hline(yintercept = 0, color = "red")+
  coord_flip()+
  xlab("")+
  ylab("z-value of Coefficients")+
  ggtitle("Logistic Regression")

p.rf = ggplot(rf_importance, aes(x = Variable, y = importance)) + 
  geom_violin()+
  geom_boxplot(width = 0.1) +
  geom_hline(yintercept = 0, color = "red")+
  coord_flip()+
  xlab("")+
  ylab("Variable Importance")+
  ggtitle("Random Forest")

p.nn = ggplot(nn_importance, aes(x = Variable, y = importance)) + 
  geom_violin()+
  geom_boxplot(width = 0.1) +
  geom_hline(yintercept = 0, color = "red")+
  coord_flip()+
  xlab("")+
  ylab("Variable Importance")+
  ggtitle("Neural Network")

ggsave("Simulation_Variable_Importance_Plots.png",
       plot = grid.arrange(p.logistic,p.rf,p.nn, nrow = 1),
       device = "png",
       width = 10,
       height = 4)

shap_glm = do.call(rbind,shap_glm)
shap_ranger = do.call(rbind,shap_ranger)
shap_neural = do.call(rbind,shap_neural)

p.shapley.glm = ggplot(shap_glm, aes(x = Variable, y = Shapley)) + 
  geom_violin()+
  geom_boxplot(width = 0.1) +
  geom_hline(yintercept = 0, color = "red")+
  coord_flip()+
  xlab("")+
  ylab("mean(|Shapley Value|)")+
  ggtitle("Logistic Regression")

p.shapley.rf <- ggplot(shap_ranger, aes(x = Variable, y = Shapley)) + 
  geom_violin()+
  geom_boxplot(width = 0.1) +
  geom_hline(yintercept = 0, color = "red")+
  coord_flip()+
  xlab("")+
  ylab("mean(|Shapley Value|)")+
  ggtitle("Random Forest")

p.shapley.nn <- ggplot(shap_neural, aes(x = Variable, y = Shapley)) + 
  geom_violin()+
  geom_boxplot(width = 0.1) +
  geom_hline(yintercept = 0, color = "red")+
  coord_flip()+
  xlab("")+
  ylab("mean(|Shapley Value|)")+
  ggtitle("Neural Network")

ggsave("Simulation_Shapley_Value_Plots.png",
       plot = grid.arrange(p.shapley.glm,p.shapley.rf,p.shapley.nn, nrow = 1),
       device = "png",
       width = 10,
       height = 4)

library(purrr)
Filter(Negate(is.null), x)

mean_glm <- do.call(rbind, Filter(Negate(is.null), map(active.glm[1:B], ~ .x$mean_grad_df)))
mean_rf <- do.call(rbind, Filter(Negate(is.null), map(active.rf[1:B], ~ .x$mean_grad_df)))
mean_nn <- do.call(rbind, Filter(Negate(is.null), map(active.nn[1:B], ~ .x$mean_grad_df)))
eigenvalues_glm <- do.call(rbind, Filter(Negate(is.null), map(active.glm[1:B], ~ .x$eigenvectors_df)))
eigenvalues_rf <- do.call(rbind, Filter(Negate(is.null), map(active.rf[1:B], ~ .x$eigenvectors_df)))
eigenvalues_nn <- do.call(rbind, Filter(Negate(is.null), map(active.nn[1:B], ~ .x$eigenvectors_df)))

p.mean_glm <- ggplot(mean_glm, 
                     aes(x = Variable, y = MeanGrad)) + 
  geom_violin()+
  geom_boxplot(width = 0.1) +
  geom_hline(yintercept = 0, color = "red")+
  coord_flip()+
  xlab("")+
  ylab("mean(Gradient)")

p.eigenvalues1_glm <- ggplot(eigenvalues_glm, 
                             aes(x = Variable, y = PC1)) + 
  geom_violin()+
  geom_boxplot(width = 0.1) +
  geom_hline(yintercept = 0, color = "red")+
  coord_flip()+
  xlab("")+
  ylab("PC 1 of Gradients")

p.eigenvalues2_glm <- ggplot(eigenvalues_glm, 
                             aes(x = Variable, y = PC2)) + 
  geom_violin()+
  geom_boxplot(width = 0.1) +
  geom_hline(yintercept = 0, color = "red")+
  coord_flip()+
  xlab("")+
  ylab("PC 2 of Gradients")

p.mean_rf <- ggplot(mean_rf, 
                    aes(x = Variable, y = MeanGrad)) + 
  geom_violin()+
  geom_boxplot(width = 0.1) +
  geom_hline(yintercept = 0, color = "red")+
  coord_flip()+
  xlab("")+
  ylab("mean(Gradient)")

p.eigenvalues1_rf <- ggplot(eigenvalues_rf, 
                            aes(x = Variable, y = PC1)) + 
  geom_violin()+
  geom_boxplot(width = 0.1) +
  geom_hline(yintercept = 0, color = "red")+
  coord_flip()+
  xlab("")+
  ylab("PC 1 of Gradients")

p.eigenvalues2_rf <- ggplot(eigenvalues_rf, 
                            aes(x = Variable, y = PC2)) + 
  geom_violin()+
  geom_boxplot(width = 0.1) +
  geom_hline(yintercept = 0, color = "red")+
  coord_flip()+
  xlab("")+
  ylab("PC 2 of Gradients")

p.mean_nn <- ggplot(mean_nn, 
                    aes(x = Variable, y = MeanGrad)) + 
  geom_violin()+
  geom_boxplot(width = 0.1) +
  geom_hline(yintercept = 0, color = "red")+
  coord_flip()+
  xlab("")+
  ylab("mean(Gradient)")

p.eigenvalues1_nn <- ggplot(eigenvalues_nn, 
                            aes(x = Variable, y = PC1)) + 
  geom_violin()+
  geom_boxplot(width = 0.1) +
  geom_hline(yintercept = 0, color = "red")+
  coord_flip()+
  xlab("")+
  ylab("PC 1 of Gradients")

p.eigenvalues2_nn <- ggplot(eigenvalues_nn, 
                            aes(x = Variable, y = PC2)) + 
  geom_violin()+
  geom_boxplot(width = 0.1) +
  geom_hline(yintercept = 0, color = "red")+
  coord_flip()+
  xlab("")+
  ylab("PC 2 of Gradients")

ggsave("Simulation2_Mean_Gradient_Plots.png",
       plot = grid.arrange(p.mean_glm,p.mean_rf,p.mean_nn, nrow = 1, ncol = 3),
       device = "png",
       width = 10,
       height = 4)

ggsave("Simulation2_Eigenvalues1_Plots.png",
       plot = grid.arrange(p.eigenvalues1_glm,p.eigenvalues1_rf,p.eigenvalues1_nn, nrow = 1),
       device = "png",
       width = 10,
       height = 4)

ggsave("Simulation2_Eigenvalues2_Plots.png",
       plot = grid.arrange(p.eigenvalues2_glm,p.eigenvalues2_rf,p.eigenvalues2_nn, nrow = 1),
       device = "png",
       width = 10,
       height = 4)

# ggplot a scatterplot of 1st column and second column with color given by pred_prob
library(ggplot2)

ggplot(GradMatrix, aes(x = GradMatrix[, 1], y = GradMatrix[, 2], color = pred_prob)) +
  geom_point() +
  scale_color_gradient(low = "blue", high = "red") +
  theme_minimal() +
  labs(x = "Column 1", y = "Column 2", color = "Prediction Probability")

# Plot gradMatrix 1st column and 2nd column with color given by pred_prob.
plot(GradMatrix[, 1], GradMatrix[, 2], col = pred_probs)

# Add on the gradient plot and arrow that gives PC1's 1st and 2nd coordinate.
# + geom_segment(aes(x = 0, y = 0, xend = abs(PC1[1]), yend = abs(PC2[1]),
# arrow = arrow(length = unit(0.5, "cm")))

misclass = data.frame()
for(i in 1:B){
  misclass_error = c(glm.error[i], rf.error[i], nn.error[i])
  misclass =rbind(misclass, misclass_error)
}
colnames(misclass) <- c("glm.error", "rf.error", "nn.error")
misclass


for(i in 1:B){
  active.glm[[i]] = active_subspace.plot(data = test[,1:d], pred_probs = fit.simulation$logistic)
  
  # Random Forest
  active.rf[[i]] = active_subspace.plot(data = test[,1:d], pred_probs = fit.simulation$randomForest)
  
  # Neural Network
  active.nn[[i]] = active_subspace.plot(data = test[,1:d], pred_probs = fit.simulation$neuralNetwork)
}

for(i in 1:B){
  plot(eigenvalues_glm[, 1], eigenvalues_glm[, 2], col = fit.simulation$logistic)
  plot(eigenvalues_rf[,1], eigenvalues_rf[,2], col = fit.simulation$randomForest)
  plot(eigenvalues_nn[,1], eigenvalues_nn[,2], col = fit.simulation$neuralNetwork)
  }

