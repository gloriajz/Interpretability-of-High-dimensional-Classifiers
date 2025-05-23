rm(list = ls())
source("functions.R")

# Load dataset

credit_risk_dataset <- read.csv("credit_risk_dataset.csv")
credit_risk <- data.frame(credit_risk_dataset)

# Deal with missing values

credit_risk[credit_risk == "NA"] = NA
credit_risk_no_missing <- credit_risk[complete.cases(credit_risk), ]

# Convert categorical variables to factors
credit_risk_no_missing$person_home_ownership <- as.factor(credit_risk_no_missing$person_home_ownership)
credit_risk_no_missing$loan_intent <- as.factor(credit_risk_no_missing$loan_intent)
credit_risk_no_missing$loan_grade <- as.factor(credit_risk_no_missing$loan_grade)
credit_risk_no_missing$cb_person_default_on_file <- as.factor(credit_risk_no_missing$cb_person_default_on_file)

# Construct training and testing datasets

samp = sample(nrow(credit_risk_no_missing), floor(nrow(credit_risk_no_missing)/2))
credit_risk_train = credit_risk_no_missing[samp,]
credit_risk_test = credit_risk_no_missing[-samp,]
train = credit_risk_train
test = credit_risk_test
colnames(train)[9] = "y"
colnames(test)[9] = "y"

# Data visualisation
ggpairs(credit_risk_no_missing, aes(col = as.character(loan_status)))+theme_bw()
ggsave("Credit Risk Data Visualisation.png",
       plot = ggpairs(credit_risk_no_missing, aes(col = Species))+theme_bw(),
       device = "png",
       width = 10,
       height = 4)

# Fit Classifiers

fit.credit_risk = fit.classifiers(train, test)

# Misclassification error

true = as.numeric(test$y)
glm.error = misclassificationerror(fit.credit_risk$logistic, true, threshold = 0.5)
rf.error = misclassificationerror(fit.credit_risk$randomForest, true, threshold = 0.5)
nn.error = misclassificationerror(fit.credit_risk$neuralNetwork, true, threshold = 0.5)

# Variable Importance plots
plots = variable.importance.plots(fit.credit_risk)

library(forcats)
p.logistic = ggplot(plots$logistic_importance, aes(reorder(Variable, importance), importance)) + 
  geom_col()+
  coord_flip()+
  ylim(c(-20, 70))+
  geom_text(aes(label = round(importance,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("z-value of Coefficients")+
  ggtitle("Logistic Regression")
p.logistic = p.logistic + aes(x = fct_inorder(Variable))

p.rf = ggplot(plots$rf_importance, aes(reorder(Variable, importance), importance)) + 
  geom_col()+
  coord_flip()+
  ylim(c(-5, 2000))+
  geom_text(aes(label = round(importance,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("Variable Importance")+
  ggtitle("Random Forest")
p.rf = p.rf + aes(x = fct_inorder(Variable))

p.nn = ggplot(plots$nn_importance, aes(reorder(Variable, importance), importance)) + 
  geom_col()+
  coord_flip()+
  ylim(c(-120,350))+
  geom_text(aes(label = round(importance,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("Variable Importance")+
  ggtitle("Neural Network")
p.nn = p.nn + aes(x = fct_inorder(Variable))

ggsave("Credit_Risk_Variable_Importance_Plots.png",
       plot = grid.arrange(p.logistic,p.rf,p.nn, nrow = 1),
       device = "png",
       width = 10,
       height = 4)

# NN plot

nn.plot(fit.credit_risk)

# Shapley Plots
# Logistic Regression Plot

shap_glm = Shapley_value.plots(fit.credit_risk$logistic_model, test = test[, -9], 
                               pred_wrapper = pfun_glm, exact = TRUE)

p.shapley.glm <- ggplot(shap_glm, aes(reorder(Variable, Shapley), Shapley)) + 
  geom_col()+
  coord_flip()+
  ylim(c(-2, 35000))+
  geom_text(aes(label = round(Shapley,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("mean(|Shapley Value|)")+
  ggtitle("Logistic Regression")
p.shapley.glm = p.shapley.glm + aes(x = fct_inorder(Variable))

# Random Forest Plot
y.colID
shap_ranger = Shapley_value.plots(fit.credit_risk$rf_model, test = test[, -9], 
                                  pred_wrapper = pfun_ranger, nsim = 10)

p.shapley.rf <- ggplot(shap_ranger, aes(reorder(Variable, Shapley), Shapley)) + 
  geom_col()+
  coord_flip()+
  ylim(c(-2, 3000))+
  geom_text(aes(label = round(Shapley,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("mean(|Shapley Value|)")+
  ggtitle("Random Forest")
p.shapley.rf = p.shapley.rf + aes(x = fct_inorder(Variable))

# Neural Network Plot

shap_neural = Shapley_value.plots(fit.credit_risk$neural_model, test = test.nn.matrix, 
                                  pred_wrapper = pfun_neural, nsim = 10)

p.shapley.nn <- ggplot(shap_neural, aes(reorder(Variable, Shapley), Shapley)) + 
  geom_col()+
  coord_flip()+
  ylim(c(0, 5000))+
  geom_text(aes(label = round(Shapley,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("mean(|Shapley Value|)")+
  ggtitle("Neural Network")
p.shapley.nn = p.shapley.nn + aes(x = fct_inorder(Variable))

ggsave("Credit_Risk_Shapley_Value_Plots.png",
       plot = grid.arrange(p.shapley.glm,p.shapley.rf,p.shapley.nn, nrow = 1),
       device = "png",
       width = 10,
       height = 4)

# Active Subspace Plots

# Mean gradient and Eigenvalues plot

# Logistic Regression
active.glm = active_subspace.plot(data = test[, -9], pred_probs = fit.credit_risk$logistic)
mean_glm <- ggplot(active.glm$mean_grad_df, 
                   aes(reorder(Variable, MeanGrad), MeanGrad)) + 
  geom_col()+
  coord_flip()+
  ylim(c(-0.05, 0.5))+
  geom_text(aes(label = round(MeanGrad,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("mean(Gradient)")
mean_glm = mean_glm + aes(x = fct_inorder(Variable))

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
  ylim(c(-1, 1.5))+
  geom_text(aes(label = round(PC1,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("PC 2 of Gradients")
eigenvalues2_glm = eigenvalues2_glm + aes(x = fct_inorder(Variable))

# Random Forest
active.rf = active_subspace.plot(data = test[, !names(test) %in% "y"], pred_probs = fit.credit_risk$randomForest)
mean_rf <- ggplot(active.rf$mean_grad_df, 
                  aes(reorder(Variable, MeanGrad), MeanGrad)) + 
  geom_col()+
  coord_flip()+
  ylim(c(-0.02, 0.2))+
  geom_text(aes(label = round(MeanGrad,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("mean(Gradient)")
mean_rf = mean_rf + aes(x = fct_inorder(Variable))

eigenvalues1_rf <- ggplot(active.rf$eigenvectors_df, 
                          aes(reorder(Variable, PC1), PC1)) + 
  geom_col()+
  coord_flip()+
  ylim(c(-0.5, 1))+
  geom_text(aes(label = round(PC1,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("PC 1 of Gradients")
eigenvalues1_rf = eigenvalues1_rf + aes(x = fct_inorder(Variable))

eigenvalues2_rf <- ggplot(active.rf$eigenvectors_df, 
                          aes(reorder(Variable, PC2), PC2)) + 
  geom_col()+
  coord_flip()+
  ylim(c(-1, 1.))+
  geom_text(aes(label = round(PC1,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("PC 2 of Gradients")
eigenvalues2_rf = eigenvalues2_rf + aes(x = fct_inorder(Variable))

# Neural Network
active.nn = active_subspace.plot(data = test[,1:4], pred_probs = fit.credit_risk$neuralNetwork)
mean_nn <- ggplot(active.nn$mean_grad_df, 
                  aes(reorder(Variable, MeanGrad), MeanGrad)) + 
  geom_col()+
  coord_flip()+
  ylim(c(-0.01, 0.01))+
  geom_text(aes(label = round(MeanGrad,2)), hjust = -0.2, colour = "red")+
  xlab("")+
  ylab("mean(Gradient)")
mean_nn = mean_nn + aes(x = fct_inorder(Variable))

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

ggsave("Mean_Gradient_Plots.png",
       plot = grid.arrange(mean_glm,mean_rf,mean_nn, nrow = 1),
       device = "png",
       width = 10,
       height = 4)

ggsave("Eigenvalues1_Plots.png",
       plot = grid.arrange(eigenvalues1_glm,eigenvalues1_rf,eigenvalues1_nn, nrow = 1),
       device = "png",
       width = 10,
       height = 4)

ggsave("Eigenvalues2_Plots.png",
       plot = grid.arrange(eigenvalues2_glm,eigenvalues2_rf,eigenvalues2_nn, nrow = 1),
       device = "png",
       width = 10,
       height = 4)