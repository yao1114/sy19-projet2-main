
library(gridExtra)
library(ggplot2)
library(caret)
library(leaps)
library(MASS)
library(glmnet)
library(gam)


# ---------------------------------------------
setwd("/lxhome/hohoang/sy19-projet2/communities")
communities <- read.csv("../Data/communities_train.csv", sep = ",", header = TRUE)
length(communities)
dim(communities)[1]

# * Remove county and community +  columns from LemasSwornFT -> PolicAveOTWorked +
# * PolicCars due to having lots of NAs
communities <- communities[, colSums(is.na(communities)) == 0]

# * We also see that the tuple (state , communityname) is categorical
# * and unique every row -> not useful in training
communities <- subset(communities, select = -c(state, communityname))
summary(communities)


#------Prétraitement：Principle component analysis--------#
# *** https://stats.stackexchange.com/questions/405660/pca-in-production-use-with-incoming-data
library(factoextra)

nb.p <- ncol(communities) - 1
res.pca <- prcomp(x = communities[, -(nb.p + 1)], center = TRUE, scale. = TRUE)
df1 <- res.pca$x
summary(res.pca)

eig <- get_eig(res.pca)
eig <- eig[1:24, ]

# Visualisation of PCA results
ggplot(data = eig, aes(x = seq_len(nrow(eig)), y = cumulative.variance.percent)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    geom_text(aes(label = round(cumulative.variance.percent, 2)), vjust = 1.6, color = "white", size = 2)
ggsave("images/pca.png")


# --------K cross validation --------#
RMSE <- function(y_test, y_predict) {
    # mean((y_test - y_predict)^2)
    # caret::RMSE(y_predict, y_test)
    sqrt(mean((y_predict - y_test)^2)) # RMSE
}

# Linear regression
linreg.cv <- function(data, p) {
    K <- 10
    n <- nrow(data)
    set.seed(1729)
    folds <- sample(1:K, n, replace = TRUE)
    err <- rep(0, K)
    for (k in 1:K) {
        train_set <- data[folds != k, ]
        test_set <- data[folds == k, ]
        fit <- lm(ViolentCrimesPerPop ~ .,
            data = train_set,
        )
        pred <- predict(fit, newdata = test_set[, -(p + 1)])
        err[k] <- RMSE(test_set[, p + 1], pred)
    }
    return(err)
}

# forward stepwise regression
fwd.stepwise.cv <- function(data, p) {
    K <- 10
    n <- nrow(data)
    set.seed(1729)
    folds <- sample(1:K, n, replace = TRUE)
    err <- rep(0, K)
    train.control <- trainControl(method = "cv", number = 10)
    for (k in 1:K) {
        train_set <- data[folds != k, ]
        test_set <- data[folds == k, ]
        fit <- caret::train(ViolentCrimesPerPop ~ .,
            data = train_set,
            method = "leapForward",
            # tuneGrid = data.frame(nvmax = 1:100),
            trControl = train.control
        )
        pred <- predict(fit, newdata = test_set[, -(p + 1)])
        err[k] <- RMSE(test_set[, p + 1], pred)
    }
    return(err)
}

bwd.stepwise.cv <- function(data, p) {
    K <- 10
    n <- nrow(data)
    set.seed(1729)
    folds <- sample(1:K, n, replace = TRUE)
    err <- rep(0, K)
    train.control <- trainControl(method = "cv", number = 10)
    for (k in 1:K) {
        train_set <- data[folds != k, ]
        test_set <- data[folds == k, ]
        fit <- caret::train(ViolentCrimesPerPop ~ .,
            data = train_set,
            method = "leapBackward",
            # tuneGrid = data.frame(nvmax = 1:100),
            trControl = train.control
        )
        pred <- predict(fit, newdata = test_set[, -(p + 1)])
        err[k] <- RMSE(test_set[, p + 1], pred)
    }
    return(err)
}

elasticnet.cv <- function(data, p) {
    K <- 10
    n <- nrow(data)
    set.seed(1729)
    folds <- sample(1:K, n, replace = TRUE)
    err <- rep(0, K)
    train.control <- trainControl(method = "cv", number = K)
    for (k in 1:K) {
        train_set <- data[folds != k, ]
        test_set <- data[folds == k, ]
        fit <- caret::train(ViolentCrimesPerPop ~ .,
            data = train_set,
            method = "glmnet",
            trControl = train.control,
            tuneLength = 10
        )
        pred <- predict(fit, newdata = test_set[, -(p + 1)])
        err[k] <- RMSE(test_set[, p + 1], pred)
    }
    return(err)
}

knn.cv <- function(data, p) {
    library(caret)
    K <- 10
    knn <- 40
    n <- nrow(data)
    set.seed(1729)
    folds <- sample(1:K, n, replace = TRUE)
    err <- rep(0, K)
    for (k in 1:K) {
        train_set <- data[folds != k, ]
        test_set <- data[folds == k, ]
        fit <- knnreg(ViolentCrimesPerPop ~ ., train_set, k = knn)
        pred <- predict(fit, test_set)
        err[k] <- RMSE(test_set[, p + 1], pred)
    }
    return(err)
}


lr.cv <- function(data, p, alpha) {
    library(glmnet)
    K <- 10
    n <- nrow(data)
    set.seed(1729)
    folds <- sample(1:K, n, replace = TRUE)
    err <- rep(0, K)
    cv.out <- cv.glmnet(as.matrix(data[, -(p + 1)]),
        data$ViolentCrimesPerPop,
        alpha = alpha,
    )

    for (k in 1:K) {
        train_set <- data[folds != k, ]
        test_set <- data[folds == k, ]
        fit <- glmnet(as.matrix(train_set[, -(p + 1)]),
            train_set$ViolentCrimesPerPop,
            lambda = cv.out$lambda.min,
            alpha = alpha,
        )
        pred <- predict(fit, newx = as.matrix(test_set[, -(p + 1)]))
        err[k] <- RMSE(test_set[, p + 1], pred)
    }
    return(err)
}



gam.cv <- function(data, p) {
    library(splines)
    library(nnet)
    K <- 10
    n <- nrow(data)
    set.seed(1729)
    folds <- sample(1:K, n, replace = TRUE)
    err <- rep(0, K)

    fm <- paste("s(", names(data)[1:p], ")", sep = "", collapse = " + ")
    fm <- as.formula(paste("ViolentCrimesPerPop ~ ", fm))

    for (k in 1:K) {
        train <- data[folds != k, ]
        test <- data[folds == k, ]
        fit <- gam(formula = fm, data = train)
        pred <- predict(fit, newdata = test[, -(p + 1)])
        err[k] <- RMSE(test[, p + 1], pred)
        message(paste(k, ""), appendLF = FALSE)
    }
    return(err)
}

bagged.cv <- function(data, p) {
    library(randomForest)
    K <- 10
    n <- nrow(data)
    set.seed(1729)
    folds <- sample(1:K, n, replace = TRUE)
    err <- rep(0, K)

    for (k in 1:K) {
        train <- data[folds != k, ]
        test <- data[folds == k, ]
        fit <- randomForest(ViolentCrimesPerPop ~ ., data = train, mtry = p)
        pred <- predict(fit, newdata = test[, -(p + 1)])
        err[k] <- RMSE(test[, p + 1], pred)
        message(paste(k, ""), appendLF = FALSE)
    }
    return(err)
}

rf.cv <- function(data, p) {
    library(randomForest)
    K <- 10
    n <- nrow(data)
    set.seed(1729)
    folds <- sample(1:K, n, replace = TRUE)
    err <- rep(0, K)

    for (k in 1:K) {
        train_set <- data[folds != k, ]
        test_set <- data[folds == k, ]
        fit <- randomForest(ViolentCrimesPerPop ~ ., data = train_set)
        pred <- predict(fit, newdata = test_set[, -(p + 1)])
        err[k] <- RMSE(test_set[, p + 1], pred)
    }
    return(err)
}


library(dplyr)
library(reticulate)
reticulate::py_config()
library(tensorflow)
library(keras)

tensorflow::tf$random$set_seed(1729)

#------Neural Networks： Multi-Layer Perceptrons------#
MLP.cv <- function(data, p, n.in) {
    K <- 10
    n <- nrow(data)
    folds <- sample(1:K, n, replace = TRUE)
    err <- rep(0, K)

    for (k in 1:K) {
        train <- data[folds != k, ]
        test <- data[folds == k, ]
        model <- keras_model_sequential()
        model %>%
            layer_dense(units = n.in, input_shape = p, activation = "relu") %>%
            layer_dense(units = n.in / 2, activation = "relu") %>%
            layer_dense(units = 1, activation = "linear")

        # compiling the defined model with metric = accuracy and optimiser as adam.
        model %>% compile(
            loss = "mse",
            optimizer = "adam",
            metrics = RMSE
        )

        model %>% fit(as.matrix(train[, -(p + 1)]), model.matrix(~ -1 + ViolentCrimesPerPop, data = train),
            epochs = 100, batch_size = 128, validation_split = 0.3
        )
        res <- model %>% evaluate(as.matrix(test[, -(p + 1)]), model.matrix(~ -1 + ViolentCrimesPerPop, data = test))
        err[k] <- res[2]
    }
    return(err)
}

#------Neural Networks：cnn------#
CNN.cv <- function(data, p, n.in) {
    K <- 10
    n <- nrow(data)
    folds <- sample(1:K, n, replace = TRUE)
    err <- rep(0, K)
    for (k in 1:K) {
        train <- data[folds != k, ]
        test <- data[folds == k, ]
        train.x <- as.matrix(train[, -(p + 1)])
        dim(train.x) <- c(nrow(train), as.integer(p / 8), 8)
        test.x <- as.matrix(test[, -(p + 1)])
        dim(test.x) <- c(nrow(test), as.integer(p / 8), 8)

        model <- keras_model_sequential()
        model %>%
            # 1 additional hidden 2D convolutional layers
            layer_dense(units = 64, activation = "relu", input_shape = c(as.integer(p / 8), 8)) %>%
            layer_max_pooling_1d(pool_size = 2) %>%
            layer_dropout(0.25) %>%
            layer_dense(units = 32, activation = "relu") %>%
            layer_flatten() %>%
            layer_dense(128) %>%
            layer_activation("relu") %>%
            layer_dropout(0.5) %>%
            layer_dense(units = 1, activation = "linear")

        model %>%
            compile(
                optimizer = "adam",
                loss = "mse",
                metrics = RMSE
            )
        model %>% fit(train.x, model.matrix(~ -1 + ViolentCrimesPerPop, data = train),
            epochs = 100, batch_size = 128, validation_split = 0.3
        )
        res <- model %>% evaluate(test.x, model.matrix(~ -1 + ViolentCrimesPerPop, data = test))
        err[k] <- res[2]
    }
    return(err)
}



#------ Linear regression, forward stepwise, backward stepwise, elastic net,
# KNN, ridge regression and lasso regression --------
nb.pca <- 22 # First 11 principal components
data.pca <- as.data.frame(res.pca$x[, 1:nb.pca])
data.pca <- cbind(data.pca, communities$ViolentCrimesPerPop)
names(data.pca)[nb.pca + 1] <- "y"

linreg.err <- linreg.cv(data.pca, nb.pca)
res <- lm(ViolentCrimesPerPop ~ ., data=communities)
summary(res)

# Remove not significative vars
library(caret)

test_sign_var <- function(model, alpha){
  sign_var <- summary(model)$coefficients[,4]
  names <- names(sign_var[sign_var<alpha])
  pred <- paste(names, collapse = "+")
  formula2 <- as.formula(paste0("ViolentCrimesPerPop", "~", pred))
  
  print(formula2)
  
  train_control <- trainControl(method = "repeatedcv",
                                number = 5,
                                repeats = 10)
  
  model <- train(formula2, data = communities,
                 trControl = train_control,
                 method = "lm")
  
  print(model$results["RMSE"]**2) # RSS train CV
}
for (alpha in c(0.1)) {
  test_sign_var(res, alpha)
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

restricted_communities <- communities[,c(
  "ViolentCrimesPerPop", 
  "racepctblack", "pctWWage", "pctWFarmSelf", 
    "whitePerCap", "PctPopUnderPov", "PctEmploy", "PctEmplManu", 
    "PctFam2Par", "PctKids2Par", "PctIlleg", "PctNotSpeakEnglWell", 
    "PctPersOwnOccup", "PctPersDenseHous", "PctHousOccup", "PctVacMore6Mos",
    "OwnOccHiQuart", "RentLowQ", "MedRentPctHousInc", "MedOwnCostPctIncNoMtg", 
    "NumStreet", "PctForeignBorn")]
names(communities)[ncol(communities)] <- "y"

mean(svr.cv(communities, ncol(communities), 10, "vanilladot", 1, 0.1))

fm <- ViolentCrimesPerPop ~ racepctblack + pctWWage + pctWFarmSelf + 
  whitePerCap + PctPopUnderPov + PctEmploy + PctEmplManu + 
  PctFam2Par + PctKids2Par + PctIlleg + PctNotSpeakEnglWell + 
  PctPersOwnOccup + PctPersDenseHous + PctHousOccup + PctVacMore6Mos + 
  OwnOccHiQuart + RentLowQ + MedRentPctHousInc + MedOwnCostPctIncNoMtg + 
  NumStreet + PctForeignBorn


formula <- ViolentCrimesPerPop ~ racepctblack + whitePerCap + PctKids2Par + PctPersDenseHous + PctVacMore6Mos + MedOwnCostPctIncNoMtg + NumStreet
res <- lm(formula, data=communities)
summary(res)

c(ViolentCrimesPerPop, racepctblack, whitePerCap, PctKids2Par, PctPersDenseHous, NumStreet)

restricted_communities <- communities[,c("ViolentCrimesPerPop", "racepctblack", "whitePerCap", "PctKids2Par", "PctPersDenseHous", "NumStreet")]

pairs(restricted_communities, verInd = 1)



fwd.stepwise.err <- fwd.stepwise.cv(data.pca, nb.pca)
bwd.stepwise.err <- bwd.stepwise.cv(data.pca, nb.pca)
elasticnet.err <- elasticnet.cv(data.pca, nb.pca)
knn.err <- knn.cv(data.pca, nb.pca)
lr.ridge.err <- lr.cv(data.pca, nb.pca, 0)
lr.lasso.err <- lr.cv(data.pca, nb.pca, 1)

# plot err
cv.nb <- rep(1:10)
classic.type <- rep(
    c(
        "linear", "forward", "backward",
        "elastic net", "KNN", "ridge", "lasso"
    ),
    each = 10
)
classic.error.value <- c(
    linreg.err, fwd.stepwise.err, bwd.stepwise.err,
    elasticnet.err, knn.err, lr.ridge.err, lr.lasso.err
)
df <- data.frame(cv.nb = cv.nb, type = classic.type, error.value = classic.error.value)
ggplot(data = df, mapping = aes(x = cv.nb, y = classic.error.value, colour = classic.type)) +
    geom_line() +
    geom_point(aes(shape = classic.type, size = 1))
ggsave("images/classic_method_err.png")


#------NB Tree + Bagging + GAM--------#
bagged.err <- bagged.cv(data.pca, nb.pca)
rf.err <- rf.cv(data.pca, nb.pca)

names(restricted_communities)[1] <- "y"
gam.cv(restricted_communities, 5, 10, deg = 1)
gam.cv(data.pca, nb.pca)

rpart.err <- rpart.cv(data.pca, nb.pca, 10)
mean(rpart.err)

prune.err <- prune.cv(data.pca, nb.pca, 10)
mean(prune.err)

library(rpart.plot)
rpart.plot(fit.prune, box.palette="RdBu", shadow.col="gray", varlen = 4,fallen.leaves=FALSE)


# plot err
ensemble.type <- rep(c("bagged", "rf", "GAM"), each = 10)
ensemble.error.value <- c(bagged.err, rf.err, gam.err)
df <- data.frame(cv.nb = cv.nb, type = ensemble.type, error.value = ensemble.error.value)
ggplot(data = df, mapping = aes(x = cv.nb, y = ensemble.error.value, colour = ensemble.type)) +
    geom_line() +
    geom_point(aes(shape = ensemble.type, size = 1))
ggsave("images/ensemble_method_err.png")

#------Neural Networks------#

nb.pca <- 16 # First 11 principal components
data.pca <- as.data.frame(res.pca$x[, 1:nb.pca])
data.pca <- cbind(data.pca, communities$ViolentCrimesPerPop)
names(data.pca)[nb.pca + 1] <- "ViolentCrimesPerPop"
MLP.err <- MLP.cv(data.pca, nb.pca, 64)
CNN.err <- CNN.cv(data.pca, nb.pca, 64)

type <- rep(c("MLP.err", "CNN.err"), each = 10)
Error.value <- c(MLP.err, CNN.err)
df <- data.frame(cv.nb = cv.nb, type = type, Error.value = Error.value)
ggplot(data = df, mapping = aes(x = cv.nb, y = Error.value, colour = type)) +
    geom_line() +
    geom_point(aes(shape = type, size = 1))
ggsave("images/dl_method_err.png")

#------Model selection------#
err.plot <- data.frame(
    linreg = linreg.err,
    fwd = fwd.stepwise.err,
    bwd = bwd.stepwise.err,
    eNet = elasticnet.err,
    KNN = knn.err,
    ridge = lr.ridge.err,
    lasso = lr.lasso.err,
    RTB = bagged.err,
    RF = rf.err,
    GAM = gam.err,
    MLP = MLP.err,
    CNN = CNN.err
)

ggplot(stack(err.plot), aes(x = ind, y = values)) +
    geom_boxplot()
ggsave("images/all_method_err.png")

# Final model - MLP model
model <- keras_model_sequential()
model %>%
    layer_dense(units = nb.pca, input_shape = nb.pca) %>%
    layer_dropout(rate = 0.8) %>%
    layer_activation(activation = "relu") %>%
    layer_dense(units = nb.class) %>%
    layer_activation(activation = "softmax")
model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = c("accuracy")
)
model %>% fit(as.matrix(data.pca[, -(nb.pca + 1)]), model.matrix(~ -1 + y, data = data.pca),
    epochs = 100, batch_size = 128, validation_split = 0.3
)
res <- model %>% evaluate(as.matrix(data.pca[, -(nb.pca + 1)]), model.matrix(~ -1 + y, data = data.pca))

summary(model)
Serr <- 1 - res[2]




Save model #
model %>% save_model_hdf5("Livrable/communities.h5")
model.serialize <- serialize_model(load_model_hdf5("keras_out.h5"))
model <- unserialize_model(model.serialize)
