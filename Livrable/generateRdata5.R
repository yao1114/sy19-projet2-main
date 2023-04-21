# 0. Chargement des données.
phoneme <- read.table('../Data/phoneme_train.txt')

robotics <- read.table('../Data/robotics_train.txt')

communities <- read.table('../Data/communities_train.csv', sep = ",", header = TRUE)

# 0.1 Transformation ACP

phoneme$y <- as.factor(phoneme$y)
phoneme.res.pca <- prcomp(x=phoneme[, -ncol(phoneme)], center=TRUE, scale.=TRUE)
phoneme.data.pca <- as.data.frame(phoneme.res.pca$x[, 1:128])
phoneme.data.pca <- cbind(phoneme.data.pca, phoneme$y)
names(phoneme.data.pca)[129] <- 'y'

communities <- communities[, colSums(is.na(communities)) == 0]
communities <- subset(communities, select = -c(state, communityname))
communities.res.pca <- prcomp(x=communities[, -ncol(communities)], center=TRUE, scale.=TRUE)
communities.data.pca <- as.data.frame(communities.res.pca$x[, 1:22])
communities.data.pca <- cbind(communities.data.pca, communities$ViolentCrimesPerPop)
names(communities.data.pca)[23] <- 'y'

# 1. Apprentissage des modèles.

## Test 4 - SVR (laplace, 22 PCA), SVR(gaussian, cubic Splines), SVM(linear, 32 PCA)

library("kernlab")
library("gam")

fm <- paste("s(", names(communities.data.pca[, 1:22]), ", df=1)", sep = "", collapse = " + ")
fm <- as.formula(paste("y ~ ", fm))
model.communities <- ksvm(fm, type="eps-svr", scaled=TRUE, sigma=0.01, 
                          kernel="laplacedot", C=10, kpar="automatic", data=communities.data.pca)

fm2 <- paste("s(", names(robotics)[-ncol(robotics)], ")", sep = "", collapse = " + ")
fm2 <- as.formula(paste("y ~ ", fm2))
model.robotics <- ksvm(fm2, data=robotics, type="eps-svr", scaled=TRUE, epsilon=0.1, 
                       kernel="rbfdot", C=100, kpar="automatic")

model.phoneme <- ksvm(y~., data=phoneme.data.pca, type="C-svc", kernel="vanilladot", C=0.01)

# 2. Création des fonctions de prédiction
prediction_phoneme <- function(dataset) {
 library("kernlab")
 dataset.data.pca <- predict(phoneme.res.pca, dataset)[, 1:128]
 predict(model.phoneme, dataset.data.pca)
}

prediction_robotics <- function(dataset) {
 library("kernlab")
 predict(model.robotics, dataset)
}

prediction_communities <- function(dataset) {
 library("gam")
 library("kernlab")
 dataset <- dataset[, colSums(is.na(dataset)) == 0]
 dataset <- subset(dataset, select = -c(state, communityname))
  
 dataset.data.pca <- predict(communities.res.pca, dataset)[, 1:22]
 predict(model.communities, dataset.data.pca)
}

# 3. Sauvegarder sous forme de fichier .Rdata les fonctions
# ’prediction_phoneme’, ’prediction_robotics’, ’prediction_communities’.
# Sauvegarder également les objets utilisés dans ces fonctions
# (‘model.phoneme‘, ‘model.robotics‘ et ‘model.communities‘ dans l’exemple) !
save(
 "communities.res.pca",
 "phoneme.res.pca",
 "model.phoneme",
 "model.robotics",
 "model.communities",
 "prediction_phoneme",
 "prediction_robotics",
 "prediction_communities",
 file = "env.Rdata"
)

