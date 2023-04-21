# 0. Chargement des données.
phoneme <- read.table('../Data/phoneme_train.txt')

robotics <- read.table('../Data/robotics_train.txt')

communities <- read.table('../Data/communities_train.csv', sep = ",", header = TRUE)

# 1. Apprentissage des modèles.

## Test 1
library("kernlab")
phoneme$y <- as.factor(phoneme$y)
model.phoneme <- ksvm(y~., data=phoneme, type="C-svc", kernel="vanilladot", C=0.01)

model.robotics <- ksvm(y~., data=robotics, type="eps-svr", scaled=TRUE, epsilon=0.1, 
                       kernel="rbfdot", C=100, kpar="automatic")

communities <- as.data.frame.array(communities)
model.communities <- ksvm(ViolentCrimesPerPop ~ ., data=communities, type="eps-svr",
                          scaled=TRUE, epsilon=0.1, 
                          kernel="laplacedot", C=10, kpar="automatic",sigma=0.01)


# 2. Création des fonctions de prédiction
prediction_phoneme <- function(dataset) {
  library("kernlab")
  predict(model.phoneme, dataset)
}

prediction_robotics <- function(dataset) {
  library("kernlab")
  predict(model.robotics, dataset)
}

prediction_communities <- function(dataset) {
  library("kernlab")
  dataset <- as.data.frame.array(dataset)
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

