'
Building, Testing, and examining Machine Learning Classification Models
'

'
=== === ===
Data Preprocessing
=== === ===
'

#Import Data
library('readr')
dataset <- read_csv('medexpense_data.csv')

#Change Smoke and Gender into binary values
SMOKER<-ifelse(dataset$smoker=="yes", 1, 0)
GENDER<-ifelse(dataset$gender=="male", 1, 0)
dataset$smoker <- SMOKER
dataset$gender <- GENDER

#Determine features with largest correlation 
install.packages("corrplot")
library(corrplot)
dataset.cor = cor(dataset)
corrplot(dataset.cor)

df = data.frame(dataset$medical_expenses, dataset$bmi, dataset$smoker)

#Encoding Target Feature
df$dataset.smoker = factor(df$dataset.smoker, levels = c(0,1))

'
=== === ===
Splitting Data and Feature Scaling
=== === ===
'
#Splitting Data into training and test data (25/75)
install.packages('caTools')
library(caTools)
set.seed(123)
split= sample.split(df$dataset.smoker, SplitRatio = 0.80)
training_set = subset(df, split==TRUE)
test_set = subset(df, split==FALSE)

# Feature Scaling
training_set[, 1:2] = scale(training_set[, 1:2])
test_set[, 1:2] = scale(test_set[, 1:2])

'
=== === ===
Training
=== === ===
'
install.packages('e1071')
install.packages('rpart')
install.packages('randomForest')
library(e1071)
library(rpart)
library(randomForest)
library(class)

#Issues with e1071: Download in Anaconda Through Environments r-e1071

#Logistic Regression
#Fitting Logistic Regression to training set and predicting results
Logistic_Classifier = glm(formula = training_set$dataset.smoker ~., family = binomial, data = training_set)

#KNN
#Fitting KNN to training set and predicting results
KNN_Classifier = knn(train = training_set[,-3], test = test_set[, -3], cl = training_set[,3], k = 10)

#SVM
#Fitting SVM to Training Set
SVM_Classifier = svm(formula = training_set$dataset.smoker ~ ., data = training_set, type = 'C-classification', kernel = 'linear')

#Kernel SVM
#Fitting Kernel SVM to Training Set
KSVM_Classifier= svm(formula = training_set$dataset.smoker ~ ., data = training_set, type = 'C-classification', kernel = 'radial')

#Naive Bayes
#Fitting Naive Bayes to Training Set
NB_Classifier= naiveBayes(x = training_set[-3], y = training_set$dataset.smoker)

#Decision Tree
#Fitting Decision Tree to Training Set
DT_Classifier = rpart(formula = training_set$dataset.smoker ~ ., data = training_set)

#Random Forest
#Fitting Decision Tree to Training Set
RF_Classifier = randomForest(x = training_set[-3], y = training_set$dataset.smoker, ntree = 100)


'
=== === ===
Model Testing
=== === ===
'
#Logistic Regression
prob_pred = predict(Logistic_Classifier, type = 'response', newdata = test_set[, -3])
y_pred_LR = ifelse(prob_pred > 0.5, 1, 0)
conf_matrix = table(test_set[,3], y_pred_LR)
conf_matrix
fourfoldplot(conf_matrix)

#KNN
y_pred_KNN = KNN_Classifier
conf_matrix = table(test_set[,3], y_pred_KNN)
conf_matrix
fourfoldplot(conf_matrix)

#SVM
y_pred_SVM = predict(SVM_Classifier, newdata = test_set[, -3])
conf_matrix = table(test_set[,3], y_pred_SVM)
conf_matrix
fourfoldplot(conf_matrix)

#K SVM
y_pred_KSVM = predict(KSVM_Classifier, newdata = test_set[, -3])
conf_matrix = table(test_set[,3], y_pred_KSVM)
conf_matrix
fourfoldplot(conf_matrix)
  
#Naive Bayes
y_pred_NB = predict(NB_Classifier, newdata = test_set[, -3])
conf_matrix = table(test_set[,3], y_pred_NB)
conf_matrix
fourfoldplot(conf_matrix)

#Decision Tree 
y_pred_DT = predict(DT_Classifier, newdata = test_set[, -3], type = 'class')
conf_matrix = table(test_set[,3], y_pred_DT)
conf_matrix
fourfoldplot(conf_matrix)
  
#Random Forest
y_pred_RF = predict(RF_Classifier, newdata = test_set[, -3])
conf_matrix = table(test_set[,3], y_pred_RF)
conf_matrix 
fourfoldplot(conf_matrix)


'
=== === ===
Data Visualization
=== === ===
'

install.packages('ElemStatLearn')
library(ElemStatLearn)

set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('dataset.medical_expenses', 'dataset.bmi')

#Logistic Regression
classifier = Logistic_Classifier
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)

plot(set[, -3], main = 'Logistic Regression (Test set)',
     xlab = 'Medical Expenses', ylab = 'BMI',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

#KNN
y_grid = knn(train = training_set[,-3], test = grid_set, cl = training_set[,3], k = 10)
plot(set[, -3], main = 'KNN (Test set)',
     xlab = 'Medical Expenses', ylab = 'BMI',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

#SVM
classifier = SVM_Classifier
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'SVM (Test set)',
     xlab = 'Medical Expenses', ylab = 'BMI',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

#K SVM
classifier = KSVM_Classifier
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'SVM (Test set)',
     xlab = 'Medical Expenses', ylab = 'BMI',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

#Naive Bayes
classifier = NB_Classifier
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'Naive Bayes',
     xlab = 'Medical Expenses', ylab = 'BMI',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

#Decision Tree
classifier = DT_Classifier
y_grid = predict(classifier, newdata = grid_set, type = 'class')
plot(set[, -3], main = 'Decision Tree (Test set)',
     xlab = 'Medical Expenses', ylab = 'BMI',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

#Random Forest
classifier = RF_Classifier
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'Random Forest (Test set)',
     xlab = 'Medical Expenses', ylab = 'BMI',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))