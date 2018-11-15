#classification: prediction which brans pf products customers prefer
#Lara Cobler Moncunill
#November 9th, 2018 - November 14th, 2018

library(readr)
library(caret)
library(rpart)
library(rpart.plot)

# Download data set
survey <- read.csv("Survey Results Complete-Table 1.csv",header=TRUE) 

#data exploration
attributes(survey)   #lists attibutes
summary(survey)      #prints min, max, median, mean, etc. each attribute
str(survey)          #displays the structure of the data set
names(survey)        #names your attributes

#change data type
survey$elevel <- ordered(survey$elevel,
                  #ordinal factor       
                        levels = c(0,1,2,3,4),
                        labels = c("less than high school","high school","some college","college dregree","Master's,doctorate or professional degree"))
survey$car <- factor(survey$car,
                     levels = c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),
                     labels = c("BMW","Buik","Cadillac","Chevrolet","Chrysler","Dodge",
                                "Ford","Honda","Hyundai","Jeep","Kia","Lincoln","Mazda",
                                "Mercedes Benz","Mitsubishi","Nissan","Ram","Subaru",
                                "Toyota","None of the above"))
survey$zipcode <- factor(survey$zipcode,
                         levels=c(0,1,2,3,4,5,6,7,8),
                         labels=c("New England","Mid-Atlantic","East North Central",
                                  "West North Central","South Atlantic","East South Central",
                                  "West South Central","Mountain","Pacific"))
survey$brand <- factor(survey$brand,
                       levels=c(0,1), labels=c("Acer","Sony"))

str(survey)
summary(survey)
sum(is.na(survey))#shows the count of NA in the dataset
#anyNA(survey)

#qqplots and histograms and for each numeric variables
for(i in 1:(ncol(survey))) {    #for every column
  if (is.numeric(survey[,i])){  #if the variable is numeric
    qqnorm(survey[,i],main=paste("Normal Q-Q Plot of",colnames(survey)[i])) #plot qqnorm
    qqline(survey[,i],col="red") #add qqnormal line in red
    hist(survey[,i],main=paste("Histogram of",colnames(survey)[i]), #make the histogram
         xlab=colnames(survey)[i])
  }
  }

#salary versus age, the only seems there is some relation with brand
plot(survey$salary,survey$age, 
     xlab="salary",ylab="age",
     col=survey$brand)
legend(x="bottomright", legend = levels(survey$brand), 
       col=1:nlevels(survey$brand), pch=1)

#because the histograms don't show a normal distribution, and we have a classification problem
#seems that the best way to focus these variables is binning them.

#binning the salary into 5 groups: low, low-medium, medium, medium-high, high
bins_salary <- 5  #number of bins
min_salary <- min(survey$salary) #minimum value of salary
max_salary <- max(survey$salary) #maximum value of salary
width_salary <- (max_salary-min_salary)/bins_salary #equal length for each group
breaks_salary <- seq(min_salary,max_salary,width_salary) #define groups seq(nim,max,interval)
labels_salary <- c("20,000-46,000","46,000-72,000","72,000-98,000","98,000-124,000","124,000-150,000")
#labels for each group
survey$salary_bin <- cut(survey$salary,breaks=breaks_salary,include.lowest = TRUE,
                         right=FALSE, labels=labels_salary)
#make a new variable in survey dataset that classifies each row to the salary group it belongs.
plot(survey$salary_bin) #plot the frequencies of the new variable

#binning the age into 3 groups: adults, middle-aged adults, senior.
bins_age <- 3
min_age <- min(survey$age)
max_age <- max(survey$age)
width_age <- (max_age-min_age)/bins_age
breaks_age <- seq(min_age,max_age,width_age)
labels_age <- c("20-40","40-60","60-80")
survey$age_bin <- cut(survey$age,breaks=breaks_age,include.lowest = TRUE,
                         right=FALSE, labels=labels_age)
plot(survey$age_bin)

#binning the credit into 5 groups: low, low-medium, medium, medium-high, high
bins_credit <- 5
min_credit <- min(survey$credit)
max_credit <- max(survey$credit)
width_credit <- (max_credit-min_credit)/bins_credit
breaks_credit <- seq(min_credit,max_credit,width_credit)
labels_credit <- c("0-1e5","1e5-2e5","2e5-3e5","3e5-4e5","4e5-5e5")
survey$credit_bin <- cut(survey$credit,breaks=breaks_credit,include.lowest = TRUE,
                         right=FALSE, labels=labels_credit)
plot(survey$credit_bin)
plot(survey$salary_bin,survey$brand)

#subset dataset only categorical variables
survey_cat <- subset(survey, select=-c(credit,salary,age))
#select the variables to remove with the "-" sign

#loop to plot all variables and brand
for(i in 1:(ncol(survey_cat))){
  table <- table(survey_cat$brand,survey_cat[,i]) #table of counts for each variable and brand
  barplot(table, main=paste("brand and",colnames(survey_cat)[i]), #bar plot of the table
          xlab=colnames(survey_cat)[i],ylab="count",
          col=c("blue","orange"),legend=rownames(table))
}
#salary seems the most important variable

#10-fold cross validation, repeated 3 times
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3,
                           classProbs=TRUE) 

#decision tree to see relationships between the variables
set.seed(123)
tree <- train(
  brand ~ ., data = survey_cat, #predict brand using all variables
  method = "rpart",
  trControl = fitControl,
  metric="Kappa",
  tuneLength = 10)
rpart.plot(tree$finalModel) #plot the decision tree
prp(tree$finalModel) #also plots the decision tree
#salary and age the most important variables!

# Predictions using salary and age, and brand because seems to be the most inflent variables.

set.seed(333)
# define an 75%/25% train/test split of the dataset
inTraining <- createDataPartition(survey_cat$brand, p = .75, list = FALSE)
#creates a vector with the rows to use for training
training <- survey_cat[inTraining,] #subset training set
testing <- survey_cat[-inTraining,] #subset testing set

#train KNN, tunning parameter k
set.seed(123)
knn_SAB <- train(
  brand ~ salary_bin+age_bin, #use the salary and age for the prediction
  data = training,
  method = "knn", 
  preProc = c("center","scale"), #because knn method uses distances
  tuneLength = 10,
  trControl = fitControl,
  metric="Kappa") #use kappa instead of accuracy because imbalance variable
        #k-Nearest Neighbors 
        
        #7501 samples
        #2 predictor
        #2 classes: 'Acer', 'Sony' 
        
        #Pre-processing: centered (6), scaled (6) 
        #Resampling: Cross-Validated (10 fold, repeated 3 times) 
        #Summary of sample sizes: 6751, 6751, 6751, 6751, 6752, 6751, ... 
        #Resampling results across tuning parameters:
          
        #  k   Accuracy   Kappa   
        #5  0.9101438  0.810864
        #7  0.9101438  0.810864
        #9  0.9101438  0.810864
        #11  0.9101438  0.810864
        #13  0.9101438  0.810864
        #15  0.9101438  0.810864
        #17  0.9101438  0.810864
        #19  0.9101438  0.810864
        #21  0.9101438  0.810864
        #23  0.9101438  0.810864
        
        #Kappa was used to select the optimal model using the largest value.
        #The final value used for the model was k = 23.

#customize tunning grid
tuned_grid_1 <- expand.grid(k=c(1:5))
set.seed(123)
knn_SAB_1 <- train(
  brand ~ salary_bin+age_bin, #use the salary and age for the prediction
  data = training,
  method = "knn", 
  preProc = c("center","scale"), #'normalization' because knn method uses distances
  tuneGrid=tuned_grid_1,
  trControl = fitControl,
  metric="Kappa") #use kappa instead of accuracy because imbalance variable
#k-Nearest Neighbors 

#7501 samples
#2 predictor
#2 classes: 'Acer', 'Sony' 

#Pre-processing: centered (6), scaled (6) 
#Resampling: Cross-Validated (10 fold, repeated 3 times) 
#Summary of sample sizes: 6750, 6751, 6751, 6752, 6751, 6751, ... 
#Resampling results across tuning parameters:
  
#  k  Accuracy   Kappa    
#1  0.9101448  0.8108757
#2  0.9101448  0.8108757
#3  0.9101448  0.8108757
#4  0.9101448  0.8108757
#5  0.9101448  0.8108757

#Kappa was used to select the optimal model using the largest value.
#The final value used for the model was k = 5.

#all the same, we can use the knn_SAB model
#maybe that all are the same because we have the same samples of all values,
#no mather the neigtbours you check it's the same.

#because is oredicting the distance between the same group have the same values,
#try k=1000 k=2000, k=3000, k=4000
#groups of 3300 for age and 2000 for salary
# tuned_grid_2 <- expand.grid(k=c(1000,2000,3000,4000)) #crash RStudio

predictors(knn_SAB)
summary(knn_SAB)
ggplot(knn_SAB)

#make predictions
test_knn_SAB <- predict(knn_SAB, newdata=testing)
knn_prob_SAB <- predict(knn_SAB,newdata=testing,type="prob") #with probabilities

#confusion matrix
confusionMatrix(data=test_knn_SAB, testing$brand)
        #Confusion Matrix and Statistics
        
        #Reference
        #Prediction Acer Sony
        #Acer  854  133
        #Sony   91 1421
        
        #Accuracy : 0.9104          
        #95% CI : (0.8985, 0.9213)
        #No Information Rate : 0.6218          
        #P-Value [Acc > NIR] : < 2.2e-16       
        
        #Kappa : 0.8111          
        #Mcnemar's Test P-Value : 0.006155        
        
        #Sensitivity : 0.9037          
        #Specificity : 0.9144          
        #Pos Pred Value : 0.8652          
        #Neg Pred Value : 0.9398          
        #Prevalence : 0.3782          
        #Detection Rate : 0.3417          
        #Detection Prevalence : 0.3950          
        #Balanced Accuracy : 0.9091          
        
        #'Positive' Class : Acer            

#performace measurment
postResample(test_knn_SAB, testing$brand)
      #> postResample(test_knn_SAB, testing$brand)
      #Accuracy     Kappa 
      #0.9103641 0.8110549

#plot predicted verses real
plot(test_knn_SAB,testing$brand)

## Random forest prediction, tunning parameter mtry
set.seed(123)
rf_SAB<- train(brand~salary_bin+age_bin, data=training, method="rf", 
               tuneLength=10,
               trControl=fitControl)
        #Random Forest 
        
        #7501 samples
        #32 predictor
        #2 classes: 'Acer', 'Sony' 
        
        #No pre-processing
        #Resampling: Cross-Validated (10 fold, repeated 3 times) 
        #Summary of sample sizes: 6751, 6752, 6752, 6751, 6750, 6750, ... 
        #Resampling results across tuning parameters:
          
        #  mtry  Accuracy   Kappa    
        #2     0.8570410  0.6825911
        #3     0.9101449  0.8108638
        #4     0.9101449  0.8108638
        #5     0.9101449  0.8108638
        #6     0.9101449  0.8108638
        
        #Accuracy was used to select the optimal model using the largest value.
        #The final value used for the model was mtry = 3.

#try different number of trees
model_RF <- list() #create a list of models
for (ntree in c(10, 20, 50, 100)) {  #for each number of trees
  set.seed(123)
  fit <- train(brand~salary_bin+age_bin, data=training, method="rf", 
               metric="Kappa", tuneLength=10, trControl=fitControl, 
               ntree=ntree)
  key <- toString(ntree) #transform number of trees to string
  model_RF[[key]] <- fit #name each model and store it a variable of model_RF
}
#compare results
results_rf <- resamples(model_RF)
summary(results_rf)
dotplot(results_rf)

#all have the same accuracy and kappa -> take model_RF[[10]] because is faster

#Prediction random forest first prediction
#test_rf_SAB <- predict(rf_SAB,newdata=testing)
#rf_prob_SAB <- predict(rf_SAB,newdata=testing,type="prob")
#head(rf_prob_SAB)

#confusion matrix
#confusionMatrix(data=test_rf_SAB, testing$brand)
      #Confusion Matrix and Statistics
      
      #Reference
      #Prediction Acer Sony
      #Acer  854  133
      #Sony   91 1421
      
      #Accuracy : 0.9104          
      #95% CI : (0.8985, 0.9213)
      #No Information Rate : 0.6218          
      #P-Value [Acc > NIR] : < 2.2e-16       
      
      #Kappa : 0.8111          
      #Mcnemar's Test P-Value : 0.006155        
      
      #Sensitivity : 0.9037          
      #Specificity : 0.9144          
      #Pos Pred Value : 0.8652          
      #Neg Pred Value : 0.9398          
      #Prevalence : 0.3782          
      #Detection Rate : 0.3417          
      #Detection Prevalence : 0.3950          
      #Balanced Accuracy : 0.9091          
      
      #'Positive' Class : Acer  

#performace measurment
#postResample(test_rf_SAB, testing$brand)
#Accuracy     Kappa 
#0.9103641 0.8110549 

#plot predicted verses actual ?
#plot(test_rf_SAB,testing$brand)

#predict testing with model_RF[[10]]
treen <- toString(10)
test_rf_10 <- predict(model_RF[[treen]],newdata=testing)
rf_prob_10 <- predict(model_RF[[treen]],newdata=testing,type="prob")

#confusion matrix
confusionMatrix(data=test_rf_10, testing$brand)

postResample(test_rf_10, testing$brand)
#Accuracy     Kappa 
#0.9139656 0.8189813 

#plot predicted verses real
plot(test_rf_10,testing$brand)
      
###compare models
resamp <- resamples(list(knn=knn_SAB,rf=model_RF[[treen]]))
summary(resamp)

#paired t-test to assess wheter there is a difference in the average resampled under the ROC curve
diff <- diff(resamp)
summary(diff)
      #Call:
      #  summary.diff.resamples(object = diff)
      
      #p-value adjustment: bonferroni 
      #Upper diagonal: estimates of the difference
      #Lower diagonal: p-value for H0: difference = 0
      
      #Accuracy 
      #knn    rf        
      #knn        -5.041e-07
      #rf  0.9999           
      
      #Kappa 
      #knn    rf        
      #knn        -2.847e-05
      #rf  0.9963   
#no differences between models

##Prediction
# Download data set
survey_inc <- read.csv("SurveyIncomplete.csv",header=TRUE) 

anyNA(survey_inc) #tells if there is any NA value

hist(survey_inc$age)
hist(survey_inc$salary)
summary(survey_inc) #same range as salary, make the same partitions
#binning the salary into 5 groups: low, low-medium, medium, medium-high, high
bins_salary <- 5  #number of bins
min_salary <- min(survey_inc$salary) #minimum value of salary
max_salary <- max(survey_inc$salary) #maximum value of salary
width_salary <- (max_salary-min_salary)/bins_salary #equal length for each group
breaks_salary <- seq(min_salary,max_salary,width_salary) #define groups seq(nim,max,interval)
labels_salary <- c("20,000-46,000","46,000-72,000","72,000-98,000","98,000-124,000","124,000-150,000")
#labels for each group
survey_inc$salary_bin <- cut(survey_inc$salary,breaks=breaks_salary,include.lowest = TRUE,
                         right=FALSE, labels=labels_salary)
#make a new variable in survey dataset that classifies each row to the salary group it belongs.
plot(survey_inc$salary_bin) #plot the frequencies of the new variable

#binning the age into 3 groups: adults, middle-aged adults, senior.
bins_age <- 3 
min_age <- min(survey_inc$age)
max_age <- max(survey_inc$age)
width_age <- (max_age-min_age)/bins_age
breaks_age <- seq(min_age,max_age,width_age)
labels_age <- c("20-40","40-60","60-80")
survey_inc$age_bin <- cut(survey_inc$age,breaks=breaks_age,include.lowest = TRUE,
                      right=FALSE, labels=labels_age)
plot(survey$age_bin)

#predictions
predictions <- predict(model_RF[[treen]],newdata=survey_inc)

#add predictions to survey_inc
survey_inc <- cbind(survey_inc,predictions) #column bind prediction to dataset

#plot(survey_inc$salary,survey_inc$age,
#     main="Predicted brand related to salary and age", #title
#     ylab="Age", xlab="Salary", #x and y labels
#     type="p", #type, p as point (default)
#     col=survey_inc$predictions) #color
#legend(x="bottomright", legend = levels(survey_inc$predictions), 
#       col=1:nlevels(survey_inc$predictions), pch=1)

#jitter using ggplot
ggplot(survey, aes(salary_bin, age_bin, color = brand)) + 
  geom_jitter() +
  labs(x="Salary",
       y="Age",
       title = "Brand Preferences",
       subtitle="Complete Survey",
       color="Brand")

#predicted
ggplot(survey_inc, aes(salary_bin, age_bin, color = predictions)) + 
  geom_jitter() +
  labs(x="Salary",
       y="Age",
       title = "Predicted brand preferences",
       subtitle="Incomplete Survey",
       color="Predicted Brand")

#total brand preferences
all_table <- table(survey_inc$predictions)+table(survey$brand)
#Acer Sony 
#5731 9269 

# Pie Chart with Percentages
pie(all_table, labels = c("Acer", "Sony"), main="Brand preferences")

pct<-round(all_table/sum(all_table)*100)
lbls<-c("Acer","Sony")
lbls<-paste(lbls,pct,"%",sep=" ")
pie(all_table, labels = lbls, main="Brand preferences",col=c("blue","orange"))
