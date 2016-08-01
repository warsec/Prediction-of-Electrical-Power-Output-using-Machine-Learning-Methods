setwd(("C:/Users/Warsec/Desktop/kaggle/CCPP/"))

library(data.table)
library(caret)
library(ggplot2)
library(ggthemes)
library(neldermead)
library(scales)
library(caretEnsemble)

#read data from file
allData <- read.csv("Folds5x2_pps.csv")

summary(allData)

X <- allData[,1:4]
y <- data.frame(allData[,5])


#Add polynomial features
#Add quadratic terms
b <- length(X) + 1
for (i in 1:4) {
  for (j in i:4) {
    X[,b] <- X[,i]*X[,j]
    b <- b + 1
  }
}

#Add cubic terms
for (i in 1:4){
  for (j in i:4){
    for (k in j:4){
      X[,b] <- X[,i]*X[,j]*X[,k]
      b <- b + 1
    }
  }
}

#Add 4-degree terms
for (i in 1:4){
  for (j in i:4){
    for (k in j:4){
      for (l in k:4){
        X[,b] <- X[,i]*X[,j]*X[,k]*X[,l]
        b <- b + 1
      }
    }
  }
}

#Add 5-degree terms
for (i in 1:4){
  for (j in i:4){
    for (k in j:4){
      for (l in k:4){
        for (m in l:4){
        X[,b] <- X[,i]*X[,j]*X[,k]*X[,l]*X[,m]
        b <- b + 1
        }
      }
    }
  }
}

#Add 6-degree terms
for (i in 1:4){
  for (j in i:4){
    for (k in j:4){
      for (l in k:4){
        for (m in l:4){
          for (n in m:4){
          X[,b] <- X[,i]*X[,j]*X[,k]*X[,l]*X[,m]*X[,n]
          b <- b + 1
          }
        }
      }
    }
  }
}

#Add 7-degree terms
for (i in 1:4){
  for (j in i:4){
    for (k in j:4){
      for (l in k:4){
        for (m in l:4){
          for (n in m:4){
            for (o in n:4){
            X[,b] <- X[,i]*X[,j]*X[,k]*X[,l]*X[,m]*X[,n]*X[,o]
            b <- b + 1
            }
          }
        }
      }
    }
  }
}

#Add 8-degree terms
for (i in 1:4){
  for (j in i:4){
    for (k in j:4){
      for (l in k:4){
        for (m in l:4){
          for (n in m:4){
            for (o in n:4){
              for (p in o:4){
              X[,b] <- X[,i]*X[,j]*X[,k]*X[,l]*X[,m]*X[,n]*X[,o]*X[,p]
              b <- b + 1
              }
            }
          }
        }
      }
    }
  }
}

# #Scaling the features
# X <- data.frame(lapply(X, rescale))

#PCA
PC <- prcomp(X, scale. = T, center = T)
PComp <- PC$x[,1:10]


#Creating train and test data
trainIndex <- createDataPartition(X$AT, p = 0.75,list = FALSE)
Xtrain <- PComp[trainIndex,]
ytrain <- y[trainIndex,]
Xtest <- PComp[-trainIndex,]
ytest <- y[-trainIndex,]

train <- cbind(Xtrain, ytrain)
test <- cbind(Xtest, ytest)

trainControl <- trainControl(method = 'repeatedcv', number = 10)

lm_fit <- train(ytrain ~ ., data = train,
                method = 'lm',
                trControl = trainControl)

Bstlm_fit <- train(ytrain ~., data = train,
                   method = 'BstLm',
                   trControl = trainControl)

enet_fit <- train(ytrain ~., data = train,
                   method = 'enet',
                   trControl = trainControl)

glmnet_fit <- train(ytrain ~., data = train,
                  method = 'glmnet',
                  trControl = trainControl)

bayes_fit <- train(ytrain ~., data = train,
                    method = 'bridge',
                    trControl = trainControl)

pls_fit <- train(ytrain ~., data = train,
                   method = 'kernelpls',
                   trControl = trainControl)

ridge_fit <- train(ytrain ~., data = train,
                   method = 'ridge',
                   trControl = trainControl)

rvm_fit <- train(ytrain ~., data = train,
                   method = 'rvmLinear',
                   trControl = trainControl)

rlm_fit <- train(ytrain ~., data = train,
                 method = 'rlm',
                 trControl = trainControl)

svm_fit <- train(ytrain ~., data = train,
                 method = 'svmLinear',
                 trControl = trainControl)

lasso_fit <- train(ytrain ~., data = train,
                 method = 'lasso',
                 trControl = trainControl)

ensemble_results <- resamples(list(Linear = lm_fit,
                                   Boosted.Linear = Bstlm_fit,
                                   Elastic.Net = enet_fit,
                                   GlmNet = glmnet_fit,
                                   Bayesian.Ridge = bayes_fit,
                                   Partial.Least.Squares = pls_fit,
                                   Ridge = ridge_fit,
                                   Robust.Linear = rlm_fit,
                                   Support.Vector = svm_fit,
                                   Lasso = lasso_fit))
summary(ensemble_results)

dotplot(ensemble_results)

#Predict
lm_predict <- predict(lm_fit, Xtest)
rlm_predict <- predict(rlm_fit, Xtest)

RMSE(lm_predict, ytest)  #4.4151
RMSE(rlm_predict, ytest)  #4.421


