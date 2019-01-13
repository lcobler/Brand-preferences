#classification: prediction which product brand customers prefer
#Lara Cobler Moncunill
#November 9th, 2018 - November 14th, 2018

library(readr)
library(caret)
library(rpart)
library(rpart.plot)
library(dplyr)

# Download data set
survey <- read.csv("/Users/lara/Dropbox/Ubiqum/Lesson 2/Task 2/Survey Results Complete-Table 1.csv",header=TRUE) 

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

#salary versus age, the only that seems there is some relation with brand
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

# Predictions using salary and age, and brand because seems to be the most influent variables.

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

#all the same, we can use the knn_SAB model
#because is oredicting the distance between the same group have the same values,

predictors(knn_SAB)
summary(knn_SAB)
ggplot(knn_SAB)

#make predictions
test_knn_SAB <- predict(knn_SAB, newdata=testing)
knn_prob_SAB <- predict(knn_SAB,newdata=testing,type="prob") #with probabilities

#confusion matrix
confusionMatrix(data=test_knn_SAB, testing$brand)

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

#predict testing with model_RF[[10]]
treen <- toString(10)
test_rf_10 <- predict(model_RF[[treen]],newdata=testing)
rf_prob_10 <- predict(model_RF[[treen]],newdata=testing,type="prob")

#confusion matrix
confusionMatrix(data=test_rf_10, testing$brand)

postResample(test_rf_10, testing$brand)
#Accuracy     Kappa 
#0.9139656 0.8189813 

#plot predicted versus real
plot(test_rf_10,testing$brand)
      
###compare models
resamp <- resamples(list(knn=knn_SAB,rf=model_RF[[treen]]))
summary(resamp)

#paired t-test to assess wheter there is a difference in the average resampled 
diff <- diff(resamp)
summary(diff)
#no differences between models

##Prediction
# Download data set
survey_inc <- read.csv("/Users/lara/Dropbox/Ubiqum/Lesson 2/Task 2/SurveyIncomplete.csv",header=TRUE) 

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

#jitter using ggplot
ggplot(survey, aes(salary_bin, age_bin, color = brand)) + 
  geom_jitter() +
  scale_color_manual(breaks = c("Acer", "Sony"),
                     values=c("lightgreen", "steelblue"))+
  labs(x="Salary",
       y="Age",
       title = "Brand Preferences",
       subtitle="Complete Survey",
       color="Brand")

#predicted
ggplot(survey_inc, aes(salary_bin, age_bin, color = predictions)) + 
  geom_jitter() +
  scale_color_manual(breaks = c("Acer", "Sony"),
                     values=c("lightgreen", "steelblue"))+
  labs(x="Salary",
       y="Age",
       title = "Predicted brand preferences",
       subtitle="Incomplete Survey",
       color="Predicted Brand")

#total brand preferences
all_table <- table(survey_inc$predictions)+table(survey$brand)

# Pie Chart with Percentages
pie(all_table, labels = c("Acer", "Sony"), main="Brand preferences")

#pie chart with percentages
pct<-round(all_table/sum(all_table)*100)
lbls<-c("Acer","Sony")
lbls<-paste(lbls,pct,"%",sep=" ")
pie(all_table, labels = lbls, main="Brand preferences",col=c("blue","orange"))

#merge both dataset
survey_inc <- survey_inc[,c(1,2,6,7,8,9,10)]
all <- bind_rows(survey, survey_inc)
ggplot(all, aes(salary_bin, age_bin, color = brand)) + 
  geom_jitter() +
  scale_color_manual(breaks = c("Acer", "Sony"),
                     values=c("lightgreen", "steelblue"))+
  labs(x="Salary",
       y="Age",
       title = "Customer Brand Preferences",
       subtitle="All survey",
       color="Predicted Brand")
