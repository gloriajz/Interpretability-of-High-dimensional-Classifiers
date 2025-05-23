# Import necessary libraries
library(ranger)
library(caret)
library(ggplot2)
library(GGally)
library(xgboost)
library(shapr)

# Load the iris dataset
data(iris)

# Filter the dataset to include only the first two classes (Virginica and Versicolor)
iris_filtered <- iris[iris$Species %in% c("versicolor", "virginica"), ]
iris_filtered_matrix <- as.matrix(iris_filtered)

#Data visualisation 
plot(iris_filtered$Sepal.Length, iris_filtered$Sepal.Width, col = iris_filtered$Species)
ggpairs(iris_filtered, aes(col = Species))+theme_bw()

# Logistic Regression
logistic_model <- glm(as.factor(iris_filtered$Species) ~ ., data = iris_filtered, family = binomial(link = "logit"))
print(summary(logistic_model))

# Random Forest using ranger
iris_filtered$Species <- factor(iris_filtered$Species, levels = c("versicolor", "virginica"))
rf_model <- ranger(x = iris_filtered[,1:4], y = as.factor(iris_filtered$Species), 
                   importance = "impurity", probability = TRUE)
print(summary(rf_model))
print(importance(rf_model))

# Shapley value
# logistic regression
pfun <- function(glm.fit, newdata){
  return(predict(glm.fit, newdata)$predictions[,1])
}

explanation <- fastshap::explain(
  logistic_model,
  X=iris_filtered[,1:4],
  pred_wrapper = pfun,
  nsim = 10
)

# Aggregate Shapeley values
shap = data.frame(Variable = colnames(iris_filtered[,1:4]),
                  Shapley = apply(explanation, MARGIN = 2, FUN = function(x) sum(abs(x))))

ggplot(shap, aes(reorder(Variable, Shapley), Shapley)) + 
  geom_col()+
  coord_flip()+
  xlab("")+
  ylab("mean(|Shapley Value|)")

# ranger
pfun <- function(ranger.fit, newdata){
  return(predict(ranger.fit, newdata)$predictions[,1])
}

explanation <- fastshap::explain(
  rf_model,
  X=iris_filtered[,1:4],
  pred_wrapper = pfun,
  nsim = 10
)

# Aggregate Shapeley values
shap = data.frame(Variable = colnames(iris_filtered[,1:4]),
                  Shapley = apply(explanation, MARGIN = 2, FUN = function(x) sum(abs(x))))

ggplot(shap, aes(reorder(Variable, Shapley), Shapley)) + 
  geom_col()+
  coord_flip()+
  xlab("")+
  ylab("mean(|Shapley Value|)")
