# 0. Chargement des données.
phoneme <- read.table('../Data/phoneme_train.txt')

robotics <- read.table('../Data/robotics_train.txt')

communities <- read.table('../Data/communities_train.csv', sep = ",", header = TRUE)

# 0b. Pré-traitement
phoneme$y <- as.factor(phoneme$y)
phoneme.res.pca <- prcomp(x=phoneme[, -ncol(phoneme)], center=TRUE, scale.=TRUE)
phoneme.data.pca <- as.data.frame(phoneme.res.pca$x[, 1:64])
phoneme.data.pca <- cbind(phoneme.data.pca, phoneme$y)
names(phoneme.data.pca)[65] <- 'y'

communities <- communities[, colSums(is.na(communities)) == 0]
communities <- subset(communities, select = -c(state, communityname))

# 1. Apprentissage des modèles.

## Test 2: linear ksvm(fm, C=1, eps=0.1), knn(k=9), MLP (PCA = 64)
library("keras")
library("kernlab")
library("FNN")


fm <- ViolentCrimesPerPop ~ racepctblack + pctWWage + pctWFarmSelf + 
  whitePerCap + PctPopUnderPov + PctEmploy + PctEmplManu + 
  PctFam2Par + PctKids2Par + PctIlleg + PctNotSpeakEnglWell + 
  PctPersOwnOccup + PctPersDenseHous + PctHousOccup + PctVacMore6Mos + 
  OwnOccHiQuart + RentLowQ + MedRentPctHousInc + MedOwnCostPctIncNoMtg + 
  NumStreet + PctForeignBorn

model.communities <- ksvm(fm, data=communities, type="eps-svr", scaled=TRUE, epsilon=0.1, 
                          kernel="vanilladot", C=1, kpar="automatic")

model.robotics <- robotics

model.serialize <- serialize_model(load_model_hdf5("phoneme.h5"))
model.phoneme <- unserialize_model(model.serialize)

n_to_class <- function(i) {
  levels(phoneme$y)[i]
}


# 2. Création des fonctions de prédiction
prediction_phoneme <- function(dataset) {
  library("kernlab")
  # install.packages("keras")
  # library("keras")
  # install_keras()
  dataset.res.pca <- prcomp(x=dataset, center=TRUE, scale.=TRUE)
  dataset.data.pca <- as.data.frame(dataset.res.pca$x[, 1:64])

  pred <- predict(model.phoneme, as.matrix(dataset.data.pca))
  n_res <- apply(data.frame(res),1,which.max)
  class_res <- lapply(n_res, n_to_class)
  unlist(class_res)
}

prediction_robotics <- function(dataset) {
  library("FNN")
  knn.reg(model.robotics[,-ncol(model.robotics)], dataset, y=model.robotics[ncol(model.robotics)], k = 9)$pred
}

prediction_communities <- function(dataset) {
  library("kernlab")
  dataset <- dataset[, colSums(is.na(dataset)) == 0]
  dataset <- subset(dataset, select = -c(state, communityname))
  
  predict(model.communities, dataset)
}

# 3. Sauvegarder sous forme de fichier .Rdata les fonctions
# ’prediction_phoneme’, ’prediction_robotics’, ’prediction_communities’.
# Sauvegarder également les objets utilisés dans ces fonctions
# (‘model.phoneme‘, ‘model.robotics‘ et ‘model.communities‘ dans l’exemple) !
save(
  "model.phoneme",
  "model.robotics",
  "model.communities",
  "prediction_phoneme",
  "prediction_robotics",
  "prediction_communities",
  file = "env.Rdata"
)
