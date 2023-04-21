# 使用深度学习框架 Kera

# install the tensorflow
library(tensorflow)
install_tensorflow()

# install keras
library(keras)
install_keras()


# test
packageVersion('tensorflow')
packageVersion('keras')
library(tensorflow)


# ---------------------------------------------
setwd("~/Desktop/SY19/projet2")
phoneme <- read.table('phoneme_train.txt')
length(phoneme)
dim(phoneme)[1]

# draw the spectrum of phoneme
par(mfrow=c(2,1))
spectrum(phoneme[,1])
spectrum(phoneme[,2])
spectrum(phoneme[,3])
spectrum(phoneme[,4])

# data processing
phoneme$y <- as.factor(phoneme$y)
summary(phoneme$y)

#------Split Train/Test Set--------# # use cross-validation instead
#library(caret)
#set.seed(1729)
#train.index <- createDataPartition(y=phoneme$y, p=0.7, list=FALSE)
#train <- phoneme[train.index,]
#test <- phoneme[-train.index,]

#------Prétraitement：Principle component analysis--------#
library(factoextra)

nb.p <- ncol(phoneme)-1
nb.class <- nlevels(phoneme$y)
res.pca <- prcomp(x=phoneme[, -(nb.p+1)], center=TRUE, scale.=TRUE)
df1<-res.pca$x
summary(res.pca)

eig <- get_eig(res.pca)
eig[37:40,]

# Visualisation of PCA results
par(mfrow=c(1,1))
fviz_screeplot(res.pca)


# --------K cross validation --------#
# err rate of lda using cross validation (non-nested)
lda.cv <- function(data, p) {
 library("MASS")
 K <- 10
 n <- nrow(data)
 set.seed(1729)
 folds <- sample(1:K, n, replace=TRUE)
 err <- rep(0, K)
 for (k in 1:K) {
  train <- data[folds!=k,]
  test <- data[folds==k,]
  fit <- lda(y~., data=train)
  pred <- predict(fit, newdata=test[, -(p+1)])
  err[k] <- 1-mean(pred$class == test$y)   
 }
 return(err)
}

# err rate of qda using cross validation (non-nested)
qda.cv <- function(data, p) {
 library("MASS")
 K <- 10
 n <- nrow(data)
 set.seed(1729)
 folds <- sample(1:K, n, replace=TRUE)
 err <- rep(0, K)
 for (k in 1:K) {
  train <- data[folds!=k,]
  test <- data[folds==k,]
  fit <- qda(y~., data=train)
  pred <- predict(fit, newdata=test[, -(p+1)])
  err[k] <- 1-mean(pred$class == test$y)
  message(k)
 }
 return(err)
}


knn.cv <- function(data, p) {
 library(caret)
 library(kknn)
 K <- 10
 knn <- 20
 n <- nrow(data)
 set.seed(1729)
 folds <- sample(1:K, n, replace=TRUE)
 err <- rep(0, K)
 for (k in 1:K) {
  train <- data[folds!=k,]
  test <- data[folds==k,]
  fit <- kknn(y~., train, test, k = 20, distance = 5)
  pred <- fitted(fit, type = "class")
  err[k] <- 1-mean(pred == test$y)
  message(k)
 }
 return(err)
}

multinom.cv <- function(data, p) {
 library("nnet")
 K <- 10
 n <- nrow(data)
 set.seed(1729)
 folds <- sample(1:K, n, replace=TRUE)
 err <- rep(0, K)
 for (k in 1:K) {
  train <- data[folds!=k,]
  test <- data[folds==k,]
  fit <- multinom(y~., data=train)
  pred <- predict(fit, newdata=test[, -(p+1)])
  err[k] <- 1-mean(pred == test$y)
  message(k)
 }
 return(err)
}

# alpha = 0 : ridge
# alpha = 1 : lasso
lr.cv <- function(data, p, alpha) {
 library("glmnet")
 K <- 10
 n <- nrow(data)
 set.seed(1729)
 folds <- sample(1:K, n, replace=TRUE)
 err <- rep(0, K)
 cv.out <- cv.glmnet(as.matrix(data[,-(p+1)]), data$y, 
                     type.measure="class", 
                     alpha=alpha, 
                     family="multinomial")
 
 for (k in 1:K) {
  train <- data[folds!=k,]
  test <- data[folds==k,]
  fit <- glmnet(as.matrix(train[,-(p+1)]), train$y, 
                lambda=cv.out$lambda.min, 
                alpha=alpha, 
                family="multinomial")
  pred <- predict(fit, newx=as.matrix(test[, -(p+1)]), type="class")
  err[k] <- 1-mean(pred == test$y)
 }
 return(err)
}

# err rate of naive-bayes using cross validation (non-nested)
naivebayes.cv <- function(data, p) {
 library("naivebayes")
 K <- 10
 n <- nrow(data)
 set.seed(1729)
 folds <- sample(1:K, n, replace=TRUE)
 err <- rep(0, K)
 for (k in 1:K) {
  train <- data[folds!=k,]
  test <- data[folds==k,]
  fit <- naive_bayes(y~., data=train)
  pred <- predict(fit, newdata=test[, -(p+1)])
  err[k] <- 1-mean(pred == test$y)
  message(k)
 }
 return(err)
}


gam.cv <- function(data, p, nclass) {
 library("splines")
 library('nnet')
 K <- 10
 n <- nrow(data)
 set.seed(1729)
 folds <- sample(1:K, n, replace=TRUE)
 err <- rep(0, K)
 
 fm <- paste('ns(', names(data)[1:p], ')', sep = "", collapse = ' + ')
 fm <- as.formula(paste('y~ ', fm))
 
 for (k in 1:K) {
  train <- data[folds!=k,]
  test <- data[folds==k,]
  fit <- multinom(formula=fm, data=train)
  pred <- predict(fit, newdata=test[, -(p+1)])
  err[k] <- 1-mean(pred == test$y)
  message(k)
 }
 return(err)
}

rpart.cv <- function(data, p) {
 library("rpart")
 K <- 10
 n <- nrow(data)
 set.seed(1729)
 folds <- sample(1:K, n, replace=TRUE)
 err <- rep(0, K)
 
 for (k in 1:K) {
  train <- data[folds!=k,]
  test <- data[folds==k,]
  fit <- rpart(y~., data=train, 
               method = "class", 
               control = rpart.control(xval=10, minbucket=10, cp=0.00))
  pred <- predict(fit, newdata=test[, -(p+1)], type="class")
  err[k] <- 1-mean(pred == test$y)
  message(k)

 }
 return(err)
}

prune.cv <- function(data, p) {
 library("rpart")
 K <- 10
 n <- nrow(data)
 set.seed(1729)
 folds <- sample(1:K, n, replace=TRUE)
 err <- rep(0, K)
 
 for (k in 1:K) {
  train <- data[folds!=k,]
  test <- data[folds==k,]
  fit <- rpart(y~., data=train, 
               method = "class", 
               control = rpart.control(xval=10, minbucket=10, cp=0.00))
  plotcp(fit,minline=TRUE)
  
  i.min <- which.min(fit$cptable[, 4])
  cp.opt <- fit$cptable[i.min, 1]
  fit.prune <- prune(fit, cp=cp.opt)
  pred <- predict(fit.prune, newdata=test[, -(p+1)], type="class")
  err[k] <- 1-mean(pred == test$y)
  
  #library(rpart.plot)
  #rpart.plot(fit.prune, box.palette="RdBu", shadow.col="gray", varlen = 4,fallen.leaves=FALSE)
  
  message(k)
 }
 return(err)
}

bagged.cv <- function(data, p) {
 library("randomForest")
 K <- 10
 n <- nrow(data)
 set.seed(1729)
 folds <- sample(1:K, n, replace=TRUE)
 err <- rep(0, K)
 
 for (k in 1:K) {
  train <- data[folds!=k,]
  test <- data[folds==k,]
  fit <- randomForest(y~., data=train, mtry=p)
  pred <- predict(fit, newdata=test[, -(p+1)], type="response")
  err[k] <- 1-mean(pred == test$y)
  message(k)
 }
 return(err)
}

rf.cv <- function(data, p) {
 library("randomForest")
 K <- 10
 n <- nrow(data)
 set.seed(1729)
 folds <- sample(1:K, n, replace=TRUE)
 err <- rep(0, K)
 
 for (k in 1:K) {
  train <- data[folds!=k,]
  test <- data[folds==k,]
  fit <- randomForest(y~., data=train)
  pred <- predict(fit, newdata=test[, -(p+1)], type="response")
  err[k] <- 1-mean(pred == test$y)
  message(k)
 }
 return(err)
}

svm.cv.findC.linear <- function(data, p) {
 library("kernlab")
 x <- as.matrix(data[, -(p+1)])
 CC<-c(0.001,0.01,0.1,1,10,100,1000,10e4)
 N<-length(CC)
 M<-10 # nombre de répétitions de la validation croisée
 err<-matrix(0,N,M)
 for(k in 1:M){
  for(i in 1:N){
    err[i,k]<-cross(ksvm(x=x, y=data$y, type="C-svc", kernel="vanilladot",C=CC[i],cross=5))
  }
 }
 Err<-rowMeans(err)
 plot(CC,Err,type="b",log="x",xlab="C",ylab="CV error")
 return(Err)
}

svm.cv.findC.nonlinear <- function(data, p) {
 library("kernlab")
 x <- as.matrix(data[, -(p+1)])
 CC<-c(0.001,0.01,0.1,1,10,100,1000,10e4)
 N<-length(CC)
 M<-10 # nombre de répétitions de la validation croisée
 err<-matrix(0,N,M)
 for(k in 1:M){
  for(i in 1:N){
   err[i,k]<-cross(ksvm(x=x, y=data$y,type="C-svc",kernel="rbfdot",kpar="automatic",C=CC[i],cross=5))
  }
 }
 Err<-rowMeans(err)
 plot(CC,Err,type="b",log="x",xlab="C",ylab="CV error")
 return(Err)
}


svm.cv <- function(data, p, kernel, c) {
 library("kernlab")
 K <- 10
 n <- nrow(data)
 set.seed(1729)
 folds <- sample(1:K, n, replace=TRUE)
 err <- rep(0, K)
 for (k in 1:K) {
  train <- data[folds!=k,]
  test <- data[folds==k,]
  fit <- ksvm(x=as.matrix(train[, -(p+1)]), y=train$y, type="C-svc", kernel=kernel, C=c, cross=5)
  pred <- predict(fit, newdata=test[, -(p+1)])
  err[k] <- 1-mean(pred == test$y)
  message(k)
 }
 return(err)
}


library(dplyr)
library(reticulate)
reticulate::py_config()
library(tensorflow)
library(keras)


#------Neural Networks： Multi-Layer Perceptrons------#
MLP.cv <- function(data, n.in, n.out) {
 K <- 10
 n <- nrow(data)
 folds <- sample(1:K, n, replace=TRUE)
 err <- rep(0, K)
 for (k in 1:K) {
  train <- data[folds!=k,]
  test <- data[folds==k,]
  model <- keras_model_sequential()
  model %>%
   # Fully connected layer, with units representing the output latitude and input_shape representing the shape of the input tensor.
   layer_dense(units=n.in, activation="relu", input_shape=n.in) %>%
   layer_dense(units=32, activation="relu") %>%
   layer_dropout(0.6) %>%
   layer_dense(units=16, activation="relu") %>%
   layer_dropout(0.5) %>%
   # Output layer (10 numbers in total, so output latitude is 10)
   layer_dense(units=n.out, activation="softmax")
  
  #compiling the defined model with metric = accuracy and optimiser as adam.
  model %>% compile(
   loss = 'categorical_crossentropy',
   optimizer = 'adam',
   metrics = c('accuracy')
  )
  
  model %>% fit(as.matrix(train[,-(n.in+1)]), model.matrix(~ -1 + y, data=train),
                epochs=100, batch_size=128, validation_split=0.3)
  res <- model %>% evaluate(as.matrix(test[,-(n.in+1)]), model.matrix(~ -1 + y, data=test))
  err[k] <- 1-res[2]
  
 }
 return(err)
}

 #------Neural Networks：cnn------#
CNN.cv <- function(data, n.in, n.out) {
 K <- 10
 n <- nrow(data)
 folds <- sample(1:K, n, replace=TRUE)
 err <- rep(0, K)
 for (k in 1:K) {
  train <- data[folds!=k,]
  test <- data[folds==k,]
  train.x <- as.matrix(train[, -(n.in+1)])
  dim(train.x) <- c(nrow(train), as.integer(n.in/8), 8)
  test.x <- as.matrix(test[, -(n.in+1)])
  dim(test.x) <- c(nrow(test), as.integer(n.in/8), 8)
  
  model <- keras_model_sequential()
  model %>% 
   # 1 additional hidden 2D convolutional layers
   layer_conv_1d(filter = 128, kernel_size = 3, input_shape=c(as.integer(n.in/8),8)) %>%
   layer_activation("relu") %>%
   layer_conv_1d(filter = 64, kernel_size =3) %>%
   layer_activation("relu") %>%
   # Use max pooling once more
   layer_max_pooling_1d(pool_size = 2) %>%
   layer_dropout(0.25) %>%
   # 2 additional hidden 2D convolutional layers
   layer_conv_1d(filter = 64, kernel_size =3) %>%
   layer_activation("relu") %>%
   layer_conv_1d(filter = 32, kernel_size =3) %>%
   layer_activation("relu") %>%
   # Use max pooling once more
   layer_max_pooling_1d(pool_size = 2) %>%
   layer_dropout(0.25) %>%
   # Flatten max filtered output into feature vector and feed into dense layer
   layer_flatten() %>%
   layer_dense(128) %>%
   layer_activation("relu") %>%
   layer_dropout(0.5) %>%
   # Outputs from dense layer are projected onto 5 unit output layer
   layer_dense(n.out) %>%
   layer_activation("softmax")
  
  
  model %>% 
   compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = 'accuracy'
   )
  model %>% fit(train.x, model.matrix(~ -1 + y, data=train),
                epochs=100, batch_size=128, validation_split=0.3)
  res <- model %>% evaluate(test.x, model.matrix(~ -1 + y, data=test))
  err[k] <- 1-res[2]
 }
 return(err)
}



#------LDA, QDA, KNN, Logistic Regression--------#
nb.pca <- 64 # we take 64 first principal components
data.pca <- as.data.frame(res.pca$x[, 1:nb.pca])
data.pca <- cbind(data.pca, phoneme$y)
names(data.pca)[nb.pca+1] <- 'y'

lda.err <- lda.cv(data.pca, nb.pca)
qda.err <- qda.cv(data.pca, nb.pca)
knn.err <- knn.cv(data.pca, nb.pca)
mult.err <- multinom.cv(data.pca, nb.pca)
# alpha = 0 : ridge
# alpha = 1 : lasso
lr.ridge.err <- lr.cv(data.pca, nb.pca,0)
lr.lasso.err <- lr.cv(data.pca, nb.pca,1)

# plot err
cv.nb <- rep(1:10)
type <- rep(c('lda','qda','knn','Logistic Regression','ridge','lasso'),each = 10)
Error.value <- c(lda.err, qda.err, knn.err, mult.err, lr.ridge.err, lr.lasso.err)
df <- data.frame(cv.nb = cv.nb, type = type, Error.value = Error.value)
ggplot(data = df, mapping = aes(x = cv.nb, y = Error.value, colour = type)) + geom_line()


#------NB Tree & Bagging--------#
nb.pca <- 32 # we take 32 first principal components
data.pca <- as.data.frame(res.pca$x[, 1:nb.pca])
data.pca <- cbind(data.pca, phoneme$y)
names(data.pca)[nb.pca+1] <- 'y'

nb.err <- naivebayes.cv(data.pca, nb.pca)
rpart.err <- rpart.cv(data.pca, nb.pca)
prune.err <- prune.cv(data.pca, nb.pca)
bagged.err <- bagged.cv(data.pca, nb.pca)
rf.err <- rf.cv(data.pca, nb.pca)

# plot err
type <- rep(c('NaiveBayes','rpart','prune','bagged','rf'),each = 10)
Error.value <- c(nb.err, rpart.err, prune.err, bagged.err, rf.err)
df <- data.frame(cv.nb = cv.nb, type = type, Error.value = Error.value)
ggplot(data = df, mapping = aes(x = cv.nb, y = Error.value, colour = type)) + geom_line()

#------GAM------#
nb.pca <- 32 # we take 32 first principal components
data.pca <- as.data.frame(res.pca$x[, 1:nb.pca])
data.pca <- cbind(data.pca, phoneme$y)
names(data.pca)[nb.pca+1] <- 'y'

gam.err.32 <- gam.cv(data.pca, nb.pca, nb.class)
gam.err.64 <- gam.cv(data.pca, nb.pca, nb.class)
gam.err.128 <- gam.cv(data.pca, nb.pca, nb.class)

gam.err <- gam.err.32

# plot err
type <- rep(c('gam.err.32','gam.err.64','gam.err.128'),each = 10)
Error.value <- c(gam.err.32, gam.err.64, gam.err.128)
df <- data.frame(cv.nb = cv.nb, type = type, Error.value = Error.value)
ggplot(data = df, mapping = aes(x = cv.nb, y = Error.value, colour = type)) + geom_line()


#------SVM------#
nb.pca <- 256 # we test with first 32, 46, 128, 256 principal components -> chose 32
data.pca <- as.data.frame(res.pca$x[, 1:nb.pca])
data.pca <- cbind(data.pca, phoneme$y)
names(data.pca)[nb.pca+1] <- 'y'

c.optimum.linear <- svm.cv.findC.linear(data.pca, nb.pca) # find optimum c = 0.01
svm.linear.err.32 <- svm.cv(data.pca, nb.pca, "vanilladot", 0.01)
svm.linear.err.64 <- svm.cv(data.pca, nb.pca, "vanilladot", 0.01)
svm.linear.err.128 <- svm.cv(data.pca, nb.pca, "vanilladot", 0.01)
svm.linear.err.256 <- svm.cv(data.pca, nb.pca, "vanilladot", 0.01)
svm.linear.err <- svm.linear.err.32

c.optimum.gaus <- svm.cv.findC.nonlinear(data.pca, nb.pca) # find optimum c = 1
svm.gaus.err.32 <- svm.cv(data.pca, nb.pca, "rbfdot", 1)
svm.gaus.err.64 <- svm.cv(data.pca, nb.pca, "rbfdot", 1)
svm.gaus.err.128 <- svm.cv(data.pca, nb.pca, "rbfdot", 1)
svm.gaus.err.256 <- svm.cv(data.pca, nb.pca, "rbfdot", 1)
svm.gaus.err <- svm.gaus.err.32

type <- rep(c('svm.linear.err.32','svm.linear.err.256'),each = 10)
Error.value <- c(svm.linear.err.32, svm.linear.err.256)
df <- data.frame(cv.nb = cv.nb, type = type, Error.value = Error.value)
ggplot(data = df, mapping = aes(x = cv.nb, y = Error.value, colour = type)) + geom_line()

type <- rep(c('svm.linear.err.64','svm.gaus.err.64'),each = 10)
Error.value <- c(svm.linear.err.64, svm.gaus.err.64)
df <- data.frame(cv.nb = cv.nb, type = type, Error.value = Error.value)
ggplot(data = df, mapping = aes(x = cv.nb, y = Error.value, colour = type)) + geom_line()

type <- rep(c('svm.linear.err.128','svm.gaus.err.128'),each = 10)
Error.value <- c(svm.linear.err.128, svm.gaus.err.128)
df <- data.frame(cv.nb = cv.nb, type = type, Error.value = Error.value)
ggplot(data = df, mapping = aes(x = cv.nb, y = Error.value, colour = type)) + geom_line()

#------Neural Networks------#
tensorflow::tf$random$set_seed(1000)

nb.pca <- 64 # we take first 64 principal components
data.pca <- as.data.frame(res.pca$x[, 1:nb.pca])
data.pca <- cbind(data.pca, phoneme$y)
names(data.pca)[nb.pca+1] <- 'y'


MLP.err <- MLP.cv(data.pca, nb.pca, nb.class)

nb.pca <- 128 # we take first 128 principal components
data.pca <- as.data.frame(res.pca$x[, 1:nb.pca])
data.pca <- cbind(data.pca, phoneme$y)
names(data.pca)[nb.pca+1] <- 'y'

CNN.err <- CNN.cv(data.pca, nb.pca, nb.class)

type <- rep(c('MLP.err','CNN.err'),each = 10)
Error.value <- c(MLP.err, CNN.err)
df <- data.frame(cv.nb = cv.nb, type = type, Error.value = Error.value)
ggplot(data = df, mapping = aes(x = cv.nb, y = Error.value, colour = type)) + geom_line()

#------Model selection------#
Err.plot = data.frame(
 LDA = lda.err, 
 QDA = qda.err, 
 KNN = knn.err, 
 LogisticRegression = mult.err,
 Ridge = lr.ridge.err,
 Lasso = lr.lasso.err,
 NaiveBayes = nb.err,
 DecisionTree = rpart.err, 
 DecisionTreePrune = prune.err, 
 DecisionTreeBagged = bagged.err, 
 RamdomForest = rf.err,
 GAM = gam.err,
 SVMlinear = svm.linear.err, 
 SVMgaus = svm.gaus.err,
 MLP = MLP.err, 
 CNN = CNN.err
 )
boxplot(Err.plot)

# Final model - SVM lineare model
phoneme$y <- as.factor(phoneme$y)
res.pca <- prcomp(x=phoneme[, -ncol(phoneme)], center=TRUE, scale.=TRUE)
data.pca <- as.data.frame(res.pca$x[, 1:128])
data.pca <- cbind(data.pca, phoneme$y)
names(data.pca)[129] <- 'y'

library("kernlab")

model.phoneme <- ksvm(x=as.matrix(data.pca[, -33]), y=data.pca$y, type="C-svc", kernel="vanilladot", C=0.01, cross=5)
pred <- predict(model.phoneme, newdata=data.pca[, -33])
err <- 1-mean(pred == data.pca$y)
err

model.phoneme <- ksvm(y~., phoneme, type="C-svc", kernel="vanilladot", C=0.01, cross=5)
pred <- predict(model.phoneme, newdata=phoneme)
err <- 1-mean(pred == phoneme$y)
err


# Save model #
model %>% save_model_hdf5("Livrable/phoneme.h5")
model.serialize <- serialize_model(load_model_hdf5("keras_out.h5"))
model <- unserialize_model(model.serialize)

