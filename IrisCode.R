rm(list = ls())
source("functions.R")


# Set random seed

set.seed(123)

# Load dataset

iris_filtered <- iris[iris$Species %in% c("versicolor", "virginica"), ]

# Construct training and testing datasets

samp = sample(nrow(iris_filtered), floor(nrow(iris_filtered)/2))
iris_train = iris_filtered[samp,]
iris_test = iris_filtered[-samp,]
train = iris_train
test = iris_test
colnames(train)[5] = "y"
colnames(test)[5] = "y"

# Fit Classifiers

fit.iris = fit.classifiers(train, test)

# Misclassification error

true = as.numeric(test$y) - 2
glm.error = misclassificationerror(fit.iris$logistic, true, threshold = 0.5)
rf.error = misclassificationerror(fit.iris$randomForest, true, threshold = 0.5)
nn.error = misclassificationerror(fit.iris$neuralNetwork, true, threshold = 0.5)

# Variable Importance plots

plots = variable.importance.plots(fit.iris)

p.logistic = ggplot(plots$logistic_importance, aes(reorder(Variable, importance), importance)) + 
  geom_col()+
  coord_flip()+
  ylim(c(-2, 3))+
  geom_text(aes(label = round(importance,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("z-value of Coefficients")+
  ggtitle("Logistic Regression")
p.logistic = p.logistic+aes(x = fct_inorder(Variable))

p.rf = ggplot(plots$rf_importance, aes(reorder(Variable, importance), importance)) + 
  geom_col()+
  coord_flip()+
  ylim(c(0, 20))+
  geom_text(aes(label = round(importance,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("Variable Importance")+
  ggtitle("Random Forest")
p.rf = p.rf + aes(x = fct_inorder(Variable))

p.nn = ggplot(plots$nn_importance, aes(reorder(Variable, importance), importance)) + 
  geom_col()+
  coord_flip()+
  ylim(c(-270, 300))+
  geom_text(aes(label = round(importance,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("Variable Importance")+
  ggtitle("Neural Network")
p.nn = p.nn + aes(x = fct_inorder(Variable))

ggsave("Iris_Variable_Importance_Plots.png",
       plot = grid.arrange(p.logistic,p.rf,p.nn, nrow = 1),
       device = "png",
       width = 10,
       height = 4)

# NN plot

nn.plot(fit.iris)

# Shapley Plots
# Logistic Regression Plot

shap_glm = Shapley_value.plots(fit.iris$logistic_model, test = iris_test[,1:4], 
                               pred_wrapper = pfun_glm, exact = TRUE)

p.shapley.glm <- ggplot(shap_glm, aes(reorder(Variable, Shapley), Shapley)) + 
  geom_col()+
  coord_flip()+
  ylim(c(0, 420))+
  geom_text(aes(label = round(Shapley,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("mean(|Shapley Value|)")+
  ggtitle("Logistic Regression")
p.shapley.glm = p.shapley.glm + aes(x = fct_inorder(Variable))

# Random Forest Plot

shap_ranger = Shapley_value.plots(fit.iris$rf_model, test = iris_test[,1:4], 
                                  pred_wrapper = pfun_ranger, nsim = 100)

p.shapley.rf <- ggplot(shap_ranger, aes(reorder(Variable, Shapley), Shapley)) + 
  geom_col()+
  coord_flip()+
  ylim(c(0, 20))+
  geom_text(aes(label = round(Shapley,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("mean(|Shapley Value|)")+
  ggtitle("Random Forest")
p.shapley.rf = p.shapley.rf + aes(x = fct_inorder(Variable))

# Neural Network Plot

shap_neural = Shapley_value.plots(fit.iris$neural_model, test = iris_test[,1:4], 
                                  pred_wrapper = pfun_neural, nsim = 100)

p.shapley.nn <- ggplot(shap_neural, aes(reorder(Variable, Shapley), Shapley)) + 
  geom_col()+
  coord_flip()+
  ylim(c(0, 30))+
  geom_text(aes(label = round(Shapley,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("mean(|Shapley Value|)")+
  ggtitle("Neural Network")
p.shapley.nn = p.shapley.nn + aes(x = fct_inorder(Variable))

ggsave("Iris_Shapley_Value_Plots.png",
       plot = grid.arrange(p.shapley.glm,p.shapley.rf,p.shapley.nn, nrow = 1),
       device = "png",
       width = 10,
       height = 4)

# Active Subspace Plots

# Mean gradient and Eigenvalues plot

# Logistic Regression
Variable = colnames(data)
active.glm = active_subspace.plot(data = iris_test[,1:4], pred_probs = fit.iris$logistic)
p.mean_glm <- ggplot(active.glm$mean_grad_df, 
                   aes(reorder(Variable, MeanGrad), MeanGrad)) + 
  geom_col()+
  coord_flip()+
  ylim(c(-0.05, 0.2))+
  geom_text(aes(label = round(MeanGrad,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("mean(Gradient)")
p.mean_glm = p.mean_glm + aes(x = fct_inorder(Variable))

eigenvalues1_glm <- ggplot(active.glm$eigenvectors_df, 
                          aes(reorder(Variable, PC1), PC1)) + 
  geom_col()+
  coord_flip()+
  ylim(c(-1, 1.5))+
  geom_text(aes(label = round(PC1,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("PC 1 of Gradients")
eigenvalues1_glm = eigenvalues1_glm + aes(x = fct_inorder(Variable))

eigenvalues2_glm <- ggplot(active.glm$eigenvectors_df, 
                          aes(reorder(Variable, PC2), PC2)) + 
  geom_col()+
  coord_flip()+
  ylim(c(-1, 0))+
  geom_text(aes(label = round(PC1,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("PC 2 of Gradients")
eigenvalues2_glm = eigenvalues2_glm + aes(x = fct_inorder(Variable))

# Random Forest
active.rf = active_subspace.plot(data = iris_test[,1:4], pred_probs = fit.iris$randomForest)
p.mean_rf <- ggplot(active.rf$mean_grad_df, 
                   aes(reorder(Variable, MeanGrad), MeanGrad)) + 
  geom_col()+
  coord_flip()+
  ylim(c(-0.02, 0.2))+
  geom_text(aes(label = round(MeanGrad,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("mean(Gradient)")
p.mean_rf = p.mean_rf + aes(x = fct_inorder(Variable))
  
eigenvalues1_rf <- ggplot(active.rf$eigenvectors_df, 
                           aes(reorder(Variable, PC1), PC1)) + 
  geom_col()+
  coord_flip()+
  ylim(c(-0.1, 1.5))+
  geom_text(aes(label = round(PC1,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("PC 1 of Gradients")
eigenvalues1_rf = eigenvalues1_rf + aes(x = fct_inorder(Variable))

eigenvalues2_rf <- ggplot(active.rf$eigenvectors_df, 
                          aes(reorder(Variable, PC2), PC2)) + 
  geom_col()+
  coord_flip()+
  ylim(c(-1, 1.5))+
  geom_text(aes(label = round(PC1,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("PC 2 of Gradients")
eigenvalues2_rf = eigenvalues2_rf + aes(x = fct_inorder(Variable))

# Neural Network
active.nn = active_subspace.plot(data = iris_test[,1:4], pred_probs = fit.iris$neuralNetwork)
p.mean_nn <- ggplot(active.nn$mean_grad_df, 
                   aes(reorder(Variable, MeanGrad), MeanGrad)) + 
  geom_col()+
  coord_flip()+
  ylim(c(-0.01, 0.01))+
  geom_text(aes(label = round(MeanGrad,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("mean(Gradient)")
p.mean_nn = p.mean_nn + aes(x = fct_inorder(Variable))
  
eigenvalues1_nn <- ggplot(active.nn$eigenvectors_df, 
                            aes(reorder(Variable, PC1), PC1)) + 
  geom_col()+
  coord_flip()+
  ylim(c(-1, 1.5))+
  geom_text(aes(label = round(PC1,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("PC 1 of Gradients")
eigenvalues1_nn = eigenvalues1_nn + aes(x = fct_inorder(Variable))

eigenvalues2_nn <- ggplot(active.nn$eigenvectors_df, 
                          aes(reorder(Variable, PC2), PC2)) + 
  geom_col()+
  coord_flip()+
  ylim(c(-1, 1.5))+
  geom_text(aes(label = round(PC1,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("PC 2 of Gradients")
eigenvalues2_nn = eigenvalues2_nn + aes(x = fct_inorder(Variable))

ggsave("Iris_Mean_Gradient_Plots.png",
       plot = grid.arrange(p.mean_glm,p.mean_rf,p.mean_nn, nrow = 1),
       device = "png",
       width = 10,
       height = 4)

ggsave("Iris_Eigenvalues1_Plots.png",
       plot = grid.arrange(eigenvalues1_glm,eigenvalues1_rf,eigenvalues1_nn, nrow = 1),
       device = "png",
       width = 10,
       height = 4)

ggsave("Iris_Eigenvalues2_Plots.png",
       plot = grid.arrange(eigenvalues2_glm,eigenvalues2_rf,eigenvalues2_nn, nrow = 1),
       device = "png",
       width = 10,
       height = 4)
