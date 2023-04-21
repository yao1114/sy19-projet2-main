## --------------------------------- Libs installation ---------------------------------

library(tensorflow)
install_tensorflow()

library(keras)
install_keras()

library(dplyr)
library(reticulate)
reticulate::py_config()
tensorflow::tf$random$set_seed(1729)






## --------------------------------- Data load --------------------------------------------
robotics <- read.table('../Data/robotics_train.txt')
head(robotics)

#------ Split Train/Test Set --------#
set.seed(1729)
traipdex <- sample(1:4000,floor(7/10*4000))
train <- robotics[traipdex,]
test <- robotics[-traipdex,]







{## --------------------------------- Data Exploration ---------------------------------
#----- Plots -----#
pairs(robotics, verInd = 9)
# no special information expect maybe linear decreasing tendency for 4th column
plot(robotics[,9],robotics[,4])

boxplot(robotics)
# data are already scaled

#----- Correlations -----#
panel.cor <- function(x, y){
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- round(cor(x, y), digits=2)
  txt <- paste0("R = ", r)
  text(0.5, 0.5, txt, cex = 0.8)
}
pairs(robotics, panel = panel.cor)
# No real correlations

#----- PCA -----#
X<-scale(robotics)
pca<-princomp(X)
Z<-pca$scores
lambda<-pca$sdev^2
plot(cumsum(lambda)/sum(lambda),type="l",xlab="nb.cv",
     ylab="proportion of cumulative explained variance")
# Very linear increase in explained variance => all descriptors have almost the same explained var (< 0.20)

#----- Best Subset Selection -----#
library(leaps)
reg.fit <- regsubsets(y~.,data=robotics, method='exhaustive')
plot(reg.fit,scale="r2")
# r2 doesn't go very high
}







## --------------------------------- Data Analysis Functions ---------------------------------

# ----- Linear Regression ----- #
lm.cv <- function(data, p, K) {
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, p, K)
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- lm(y~., data=train)
    pred <- predict(fit, newdata=test[, -(p+1)])
    err[k] <- mean((test$y-pred)^2)   
  }
  return(err)
}

# ----- Polynomial Linear Regression ----- #
polylm.cv <- function(data, p, K, deg) {
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  
  fm <- paste('poly(', names(data)[1:p], ', deg)', sep = "", collapse = ' + ')
  fm <- as.formula(paste('y~ ', fm))
  
  err <- rep(0, p, K)
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- lm(fm, data = train)
    pred <- predict(fit, newdata = test[, 1:p])
    err[k] <- mean((test$y-pred)^2)
  }
  return(err)
}

# ----- K Nearest Neighbors ----- #
knn.cv <- function(data, p, K, kk) {
  library(FNN)
  
  knn <- 20
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, p, K)
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- knn.reg(train[,1:p], test[,1:p], y=train[,p+1], k = kk)
    pred <- fit$pred
    err[k] <- mean((test$y-pred)^2)
  }
  return(err)
}

# ----- Logistic Regression ----- #
# alpha = 0 : ridge
# alpha = 1 : lasso
lr.cv <- function(data, p, K, alpha) {
  library(glmnet)
  
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, p, K)
  cv.out <- cv.glmnet(as.matrix(data[,-(p+1)]), data$y, alpha=alpha)
  
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- glmnet(as.matrix(train[,-(p+1)]), train$y, 
                  lambda=cv.out$lambda.min, 
                  alpha=alpha)
    pred <- predict(fit, newx=as.matrix(test[, -(p+1)]), s=cv.out$lambda.min)
    err[k] <- mean((test$y-pred)^2)
  }
  return(err)
}

# ----- Natural Splines ----- #
nslm.cv <- function(data, p, K, deg) {
  library(splines)
  
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  
  fm <- paste('ns(', names(data)[1:p], ')', sep = "", collapse = ' + ')
  fm <- as.formula(paste('y~ ', fm))
  
  err <- rep(0, p, K)
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- lm(fm, data = train)
    pred <- predict(fit, newdata = test[, 1:p])
    err[k] <- mean((test$y-pred)^2)
  }
  return(err)
}

# ----- Smoothing Splines ----- #
gam.cv <- function(data, p, K, deg) {
  library(gam)
  
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, p, K)
  
  fm <- paste('s(', names(data)[1:p], ', deg)', sep = "", collapse = ' + ')
  fm <- as.formula(paste('y~ ', fm))
  
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- gam(fm, data = train)
    pred <- predict(fit, newdata = test[, 1:p])
    err[k] <- mean((test$y-pred)^2)
  }
  return(err)
}

# ----- Decision Tree ----- #
rpart.cv <- function(data, p, K) {
  library(rpart)
  
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, p, K)
  
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- rpart(y~., data=train, 
                 method = "anova", 
                 control = rpart.control(xval=10, minbucket=10, cp=0.00))
    pred <- predict(fit, newdata=test[, -(p+1)])
    err[k] <- mean((test$y-pred)^2)
  }
  return(err)
}

# ----- Decision Tree with pruning ----- #
prune.cv <- function(data, p, K) {
  library(rpart)
  
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, p, K)
  
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- rpart(y~., data=train, 
                 method = "anova", 
                 control = rpart.control(xval=10, minbucket=10, cp=0.00))
    plotcp(fit,minline=TRUE)
    
    i.min <- which.min(fit$cptable[, 4])
    cp.opt <- fit$cptable[i.min, 1]
    fit.prune <- prune(fit, cp=cp.opt)
    pred <- predict(fit.prune, newdata=test[, -(p+1)])
    err[k] <- mean((test$y-pred)^2)
    message(err[k], cp.opt)
  }
  return(err)
}

# ----- Bagging ----- #
bagging.cv <- function(data, p, K) {
  library("randomForest")
  
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, p, K)
  
  for (k in 1:K) {
    message(k)
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- randomForest(y~., data=train, mtry=p)
    pred <- predict(fit, newdata=test[, -(p+1)], type="response")
    err[k] <- mean((test$y-pred)^2)
  }
  return(err)
}

# ----- Random Forest ----- #
rf.cv <- function(data, p, K) {
  library("randomForest")
  
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, p, K)
  
  for (k in 1:K) {
    message(k)
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- randomForest(y~., data=train)
    pred <- predict(fit, newdata=test[, -(p+1)], type="response")
    err[k] <- mean((test$y-pred)^2)
  }
  return(err)
}

# ----- SVR ----- #
svr.cv.findC.linear <- function(data, p) {
  library("kernlab")
  CC<-c(0.001,0.01,0.1,1,10,100, 1000, 10e4)
  N<-length(CC)
  M<-10 # nombre de répétitions de la validation croisée
  err<-matrix(0,N,M)
  for(k in 1:M){
    for(i in 1:N){
      err[i,k]<-cross(
        ksvm(y ~., data=data, type="eps-svr", scaled=TRUE, epsilon=0.1,
             kernel="vanilladot",C=CC[i]))
    }
  }
  Err<-rowMeans(err)
  plot(CC,Err,type="b",log="x",xlab="C",ylab="CV error")
  return(Err)
}

svr.cv.findC.nonlinear <- function(data, p) {
  library("kernlab")
  x <- as.matrix(data[, -(p+1)])
  CC<-c(0.001,0.01,0.1,1,10,100)
  N<-length(CC)
  M<-1 # nombre de répétitions de la validation croisée
  err<-matrix(0,N,M)
  for(k in 1:M){
    for(i in 1:N){
      err[i,k]<-cross(
        ksvm(y ~., data=data, type="eps-svr", scaled=TRUE, epsilon=0.1,
             kernel="rbfdot",C=CC[i], kpar="automatic"))
    }
  }
  Err<-rowMeans(err)
  plot(CC,Err,type="b",log="x",xlab="C",ylab="CV error")
  return(Err)
}

svr.cv <- function(data, p, K, kernel, c, eps) {
  library("kernlab")
  
  n <- nrow(data)
  set.seed(1729)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, p, K)
  for (k in 1:K) {
    message(k)
    train <- data[folds!=k,]
    test <- data[folds==k,]
    fit <- ksvm(y~., data=train, type="eps-svr", scaled=TRUE, epsilon=eps, kernel=kernel, C=c, kpar="automatic", cross=5)
    pred <- predict(fit, newdata=test[, -(p+1)])
    err[k] <- mean((test$y-pred)^2)
    message(k)
  }
  return(err)
}

#------Neural Networks： Multi-Layer Perceptrons------#
MLP.cv <- function(data, p, K) {
  
  n <- nrow(data)
  folds <- sample(1:K, n, replace=TRUE)
  err <- rep(0, p, K)
  
  for (k in 1:K) {
    train <- data[folds!=k,]
    test <- data[folds==k,]
    model <- keras_model_sequential()
    
    # Model architecture
    model %>%
      layer_dense(units = 50, activation = 'relu', input_shape = p) %>%
      layer_dense(units = 10, activation = 'relu') %>%
      layer_dense(units = 50, activation = 'relu') %>%
      layer_dense(units = 1, activation = 'linear')
    
    # Model optimizer
    model %>% compile(
      loss = 'mean_squared_error',
      optimizer = optimizer_rmsprop()
    )
    
    model %>% fit(as.matrix(train[,-(p+1)]), as.matrix(train[,p+1]),
                  epochs=300, batch_size=512, validation_split=0.3)
    res <- model %>% evaluate(as.matrix(test[,-(p+1)]), as.matrix(test[,p+1]))
    err[k] <- res
    
  }
  return(err)
}






## --------------------------------- Data Analysis ---------------------------------
#----- CV parameters -------#
K <- 10
p <- 8 # all descriptors

#------ LM --------#
summary(lm(y~., robotics))
# all descriptors are significant, low adjusted R-squared

# Simple linear regression
lm.err <- lm.cv(robotics, p, K)
mean(lm.err)

# Polynomial linear regression
v <- c()
for (deg in seq(1,6,1)) {
  v <- append(v, mean(polylm.cv(robotics, p, K, deg)))
}

plot(seq(1,6,1), v, type = "l")
deg <- 3 #which.min(v)
# deg = 3 has the lowest error

polylm.err <- polylm.cv(robotics, p, K, deg)
mean(polylm.err)

#------ Ridge/lasso ------#
# alpha = 0 : ridge
# alpha = 1 : lasso
lr.ridge.err <- lr.cv(robotics, p, K, 0)
mean(lr.ridge.err)

lr.lasso.err <- lr.cv(robotics, p, K, 1)
mean(lr.lasso.err)

#------ Natural Splines ------#
v <- c()
for (deg in seq(1,6,1)) {
  v <- append(v, mean(nslm.cv(robotics, p, K, deg)))
}

plot(seq(1,6,1), v, type = "l")
deg <- 3 #which.min(v)
# deg = 3 has the lowest error

nslm.err <- nslm.cv(robotics, p, K, deg)
mean(nslm.err)

#------ GAM ------#
v <- c()
DEG <- seq(1,6,1)
for (deg in DEG) {
  v <- append(v, mean(gam.cv(robotics, p, K, deg)))
}

plot(DEG, v, type = "l")
deg <- 4 #which.min(v)
# deg = 4 has the lowest error

gam.err <- gam.cv(robotics, p, K, deg)
mean(gam.err)

#------ KNN ------#
v <- c()
KK <- seq(1,10,1)
for (kk in KK) {
  v <- append(v, mean(knn.cv(robotics, p, K, kk)))
}

plot(KK, v, type = "l")
kk <- 9 #which.min(v)
# kk = 9 has the lowest error

knn.err <- knn.cv(robotics, p, K, kk)
mean(knn.err)

#------ Decision trees & Random Forest --------#
rpart.err <- rpart.cv(robotics, p, K)
mean(rpart.err)

prune.err <- prune.cv(robotics, p, K)
mean(prune.err)

#library(rpart.plot)
#rpart.plot(fit.prune, box.palette="RdBu", shadow.col="gray", varlen = 4,fallen.leaves=FALSE)

bagging.err <- bagging.cv(robotics, p, K)
mean(bagging.err)

rf.err <- rf.cv(robotics, p, K)
mean(rf.err)

#------ Mixture of regressions -------#
#library(mixtools)
#mixreg <- regmixEM(train$y, as.matrix(train[,-(p+1)]), maxit = 100)
#summary(mixreg)
#plot(mixreg)
# How to predict for external value? get posterior for test?

#------ Support Vector Regression ------#
c.optimum.linear <- svr.cv.findC.linear(robotics, p) # find optimum c = 0.01
svr.linear.err <- svr.cv(robotics, p, K, "vanilladot", 0.01)
mean(svr.linear.err)

c.optimum.gaus <- svr.cv.findC.nonlinear(robotics, p) # find optimum c = 100
svr.gaus.err <- svr.cv(robotics, p, K, "rbfdot", 100, 0.2)
mean(svr.gaus.err)

#------Neural Networks------#
MLP.err <- MLP.cv(robotics, p, K)
mean(MLP.err)






## --------------------------------- Results Analysis ---------------------------------
# ----- Plot errors ----- #
library(ggplot2)
cv.nb <- rep(1:K)

# all errors
type <- rep(c('lm','polylm', 'nslm', 
              'knn','ridge','lasso', 'gam', 
              'rpart','prune','bagging','rf',
              'svrlinear', 'svrgaussian',
              'MLP'), each = 10)
Error.value <- c(lm.err, polylm.err, nslm.err, 
                 knn.err, lr.ridge.err, lr.lasso.err, gam.err,
                 rpart.err, prune.err, bagging.err, rf.err,
                 svr.linear.err, svr.gaus.err,
                 MLP.err)
df <- data.frame(cv.nb = cv.nb, type = type, Error.value = Error.value)
ggplot(data = df, mapping = aes(x = cv.nb, y = Error.value, colour = type)) + geom_line()

# lm
type <- rep(c('lm','polylm', 'nslm', 
              'ridge','lasso', 'gam'), each = 10)
Error.value <- c(lm.err, polylm.err, nslm.err, 
                 lr.ridge.err, lr.lasso.err, gam.err)
df <- data.frame(cv.nb = cv.nb, type = type, Error.value = Error.value)
ggplot(data = df, mapping = aes(x = cv.nb, y = Error.value, colour = type)) + geom_line()

# tree
type <- rep(c('rpart','prune','bagging','rf',
              'knn'), each = 10)
Error.value <- c(rpart.err, prune.err, bagging.err, rf.err,
                 knn.err)
df <- data.frame(cv.nb = cv.nb, type = type, Error.value = Error.value)
ggplot(data = df, mapping = aes(x = cv.nb, y = Error.value, colour = type)) + geom_line()

# SVR
type <- rep(c('rf','knn','linear_svr','gaussian_svr'), each = 10)
Error.value <- c(rf.err, knn.err, svr.linear.err, svr.gaus.err)
df <- data.frame(cv.nb = cv.nb, type = type, Error.value = Error.value)
ggplot(data = df, mapping = aes(x = cv.nb, y = Error.value, colour = type)) + geom_line()


# errors > 0.03
type <- rep(c('lm','polylm', 'nslm', 
              'ridge','lasso', 'gam', 
              'rpart','prune',
              'svrlinear'), each = 10)
Error.value <- c(lm.err, polylm.err, nslm.err, 
                 lr.ridge.err, lr.lasso.err, gam.err,
                 rpart.err, prune.err,
                 svr.linear.err)
df <- data.frame(cv.nb = cv.nb, type = type, Error.value = Error.value)
ggplot(data = df, mapping = aes(x = cv.nb, y = Error.value, colour = type)) + geom_line()

# errors < 0.03
type <- rep(c('knn','bagging','rf',
              'svrgaussian',
              'MLP'), each = 10)
Error.value <- c(knn.err, bagging.err, rf.err,
                 svr.gaus.err,
                 MLP.err)
df <- data.frame(cv.nb = cv.nb, type = type, Error.value = Error.value)
ggplot(data = df, mapping = aes(x = cv.nb, y = Error.value, colour = type)) + geom_line()


#------Model selection------#
Err.plot = data.frame(
  LR = lm.err, 
  PLR = polylm.err,
  Ridge = lr.ridge.err,
  Lasso = lr.lasso.err,
  NSLR = nslm.err,
  KNN = knn.err, 
  GAM = gam.err,
  Tree = rpart.err, 
  PTree = prune.err, 
  Bagging = bagging.err, 
  RF = rf.err,
  LSVR = svr.linear.err, 
  GSVR = svr.gaus.err,
  MLP = MLP.err
)

boxplot(Err.plot)

colMeans(Err.plot)
boxplot(Err.plot)




## --------------------------------- Final model ---------------------------------
# Final model - gaussian SVR model
# C = 100

model <- ksvm(y~., data=robotics, type="eps-svr", scaled=TRUE, epsilon=0.1, 
              kernel="rbfdot", C=100, kpar="automatic", cross=10)

cross(model) # 0.007


# Save keras model #
#model %>% save_model_hdf5("Livrable/robotics.h5")
#model.serialize <- serialize_model(load_model_hdf5("keras_out.h5"))
#model <- unserialize_model(model.serialize)


