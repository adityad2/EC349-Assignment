# Installing Libraries

library(jsonlite)
library(fastDummies)
library(ggplot2)
library(dplyr)
library(glmnet)
library(rpart)
library(rpart.plot)
library(ipred)
library(randomForest)
library(gbm)
library(nnet)

# Setting working directory

cat("\014")  
rm(list=ls())
setwd("C:/Warwick/EC349/Assignment")

# Parsing and saving datasets as data frames

business_data <- stream_in(file("yelp_academic_dataset_business.json"))
checkin_data  <- stream_in(file("yelp_academic_dataset_checkin.json"))
tip_data  <- stream_in(file("yelp_academic_dataset_tip.json"))

load("C:/Warwick/EC349/Assignment/yelp_review_small.Rda")
review_data <- review_data_small
rm(review_data_small)

load("C:/Warwick/EC349/Assignment/yelp_user_small.Rda")
user_data <- user_data_small
rm(user_data_small)

# Merging datasets

merged_df <- merge(review_data, business_data, by ="business_id", all.x= TRUE)
merged_df <- merge(merged_df, user_data, by ="user_id", all.x = TRUE)

# Making Dummy Variables for the states

merged_df$state <- as.factor(merged_df$state)
levels(merged_df$state)

merged_df <- dummy_cols(merged_df, select_columns = "state")

# Deciding on business attributes or user variables:

sum(rowSums(is.na(data.frame(merged_df$attributes.ByAppointmentOnly))) > 0)

sum(rowSums(is.na(data.frame(merged_df$average_stars))) > 0)

sum(rowSums(is.na(data.frame(merged_df[,c("attributes.ByAppointmentOnly","average_stars")]))) > 0)

cor(merged_df$stars.x, merged_df$average_stars, use= "complete.obs")

# Bar plot of mean stars given to businesses in different states

statewise_stars <- aggregate(stars.x ~ state, data = merged_df, mean)
barplot(statewise_stars$stars.x, names.arg = statewise_stars$state, col = "skyblue", main = "Mean Stars by State", xlab = "State", ylab = "Mean Stars")

# Bar graph of frequencies for different states

ggplot(merged_df, aes(x = state)) +
  geom_bar(fill = "lightblue", color = "black") +
  labs(title = "Bar Plot of Frequencies",
       x = "State",
       y = "Frequency in 1000s") +
  scale_y_continuous(labels = scales::comma_format(scale = 1e-3))

sum(merged_df$state == "VT")

# Choosing final Variables:

cor_matrix <- cor(merged_df[,c("stars.x", "useful.x", "funny.x", "cool.x", "stars.y", "review_count.x", "is_open", "review_count.y", "useful.y", "funny.y", "cool.y", "fans", "average_stars", "compliment_hot", "compliment_more", "compliment_profile", "compliment_cute", "compliment_list", "compliment_note", "compliment_plain", "compliment_cool", "compliment_funny", "compliment_writer", "compliment_photos")], use= "complete.obs")

cor_matrix_1 <- cor(merged_df[,c("stars.x", "useful.x", "funny.x", "cool.x")])

cor_matrix_2 <- cor(merged_df[,c("stars.x", "review_count.y", "useful.y", "funny.y", "cool.y", "fans")], use= "complete.obs")

cor_matrix_3 <- cor(merged_df[,c("stars.x","compliment_hot", "compliment_more", "compliment_profile", "compliment_cute", "compliment_list", "compliment_note", "compliment_plain", "compliment_cool", "compliment_funny", "compliment_writer", "compliment_photos")], use= "complete.obs")

# T-test to show significantly different means between businesses that are open v.s. not open

t.test(stars.x ~ is_open, data = merged_df)

# Making final merged data frame

merged_df_final <- select(merged_df, review_id, business_id, user_id, stars.x, useful.x, stars.y, review_count.x, is_open, review_count.y, average_stars, compliment_writer)
merged_df_final <- na.omit(merged_df_final)

# Making training dataset:

set.seed(1)
train <- sample(1:nrow(merged_df_final), nrow(merged_df_final) - 10000)
final_df_train <- merged_df_final[train,]

# Making test dataset:

final_df_test <- merged_df_final[-train,]

# Modelling

# Simple multivariate linear regression

lm.model <- lm(stars.x~useful.x+ stars.y+ review_count.x+ is_open+ review_count.y+ average_stars+ compliment_writer, data= final_df_train)
summary(lm.model) 

lm_predict<-predict(lm.model, newdata = final_df_test[,-4])

lm_predict_round <- data.frame(col1 = round(lm_predict))
lm_predict_round <- lm_predict_round %>%
  mutate(col2 = ifelse(col1>5,5,ifelse(col1<1, 1, col1)))
accuracy_percentage_lm <- (sum(lm_predict_round$col2 == final_df_test$stars.x)/nrow(final_df_test))*100
print(paste("Accuracy Percentage:", accuracy_percentage_lm, "%"))

matplot(1:100, cbind(lm_predict_round[1:100,2], data.frame(final_df_test$stars.x)[1:100,]), type = "l", lty = 1, 
        col = c("red", "blue"), xlab = "First 100 test data points", 
        ylab = "Stars", main = "Linear regression predictions v.s. observed stars")
legend("topright", legend = c("Prediction", "actual"), 
       col = c("red", "blue"), 
       lty = 1)

# Multinomial logistic regression

logistic.model <- multinom(as.factor(stars.x)~useful.x+ stars.y+ review_count.x+ is_open+ review_count.y+ average_stars+ compliment_writer, data = final_df_train)
logistic.pred <- predict(logistic.model, newdata = final_df_test)

logistic_predict <- data.frame(col1 = logistic.pred)
accuracy_percentage_logistic <- (sum(logistic_predict$col1 == final_df_test$stars.x)/nrow(final_df_test))*100
print(paste("Accuracy Percentage:", accuracy_percentage_logistic, "%"))


matplot(1:100, cbind(logistic_predict[1:100,], data.frame(final_df_test$stars.x)[1:100,]), type = "l", lty = 1, 
        col = c("red", "blue"), xlab = "First 100 test data points", 
        ylab = "Stars", main = "Logistic predictions v.s. observed stars")
legend("topright", legend = c("Prediction", "actual"), 
       col = c("red", "blue"), 
       lty = 1)

# Classification tree

rpart_tree_class<-rpart(as.factor(stars.x)~useful.x+ stars.y+ review_count.x+ is_open+ review_count.y+ average_stars+ compliment_writer, data=final_df_train, method= "class", cp = 0.0001)
rpart.plot(rpart_tree_class)

rpart_class.pred <- predict(rpart_tree_class, newdata = final_df_test, type= "class")

rpart_class_predict <- data.frame(rpart_class.pred)
colnames(rpart_class_predict)[1] <- "col1"
accuracy_percentage_rpart_class <- (sum(rpart_class_predict$col1 == final_df_test$stars.x)/nrow(final_df_test))*100
print(paste("Accuracy Percentage:", accuracy_percentage_rpart_class, "%"))

matplot(1:100, cbind(rpart_class_predict[1:100,], data.frame(final_df_test$stars.x)[1:100,]), type = "l", lty = 1, 
        col = c("red", "blue"), xlab = "First 100 test data points", 
        ylab = "Stars", main = "Classification Tree predictions v.s. observed stars")
legend("topright", legend = c("Prediction", "actual"), 
       col = c("red", "blue"), 
       lty = 1)

#Classification bagging

set.seed(1312)
bag_class <- bagging(as.factor(stars.x)~useful.x+ stars.y+ review_count.x+ is_open+ review_count.y+ average_stars+ compliment_writer, data = final_df_train, method= "class", nbagg = 50, coob = TRUE, control = rpart.control(minsplit = 2, cp = 0.0001))

bag_class

bag_class.pred <- predict(bag_class, newdata = final_df_test, type= "class")

bag_class_predict <- data.frame(bag_class.pred)
colnames(bag_class_predict)[1] <- "col1"
accuracy_percentage_bag_class <- (sum(bag_class_predict$col1 == final_df_test$stars.x)/nrow(final_df_test))*100
print(paste("Accuracy Percentage:", accuracy_percentage_bag_class, "%"))

matplot(1:100, cbind(bag_class_predict[1:100,], data.frame(final_df_test$stars.x)[1:100,]), type = "l", lty = 1, 
        col = c("red", "blue"), xlab = "First 100 test data points", 
        ylab = "Stars", main = "Classification Bagging predictions v.s. observed stars")
legend("topright", legend = c("Prediction", "actual"), 
       col = c("red", "blue"), 
       lty = 1)

# Classification Random Forest

set.seed(1312)
model_RF_class<-randomForest(as.factor(stars.x)~useful.x+ stars.y+ review_count.x+ is_open+ review_count.y+ average_stars+ compliment_writer, data = final_df_train, method = "class",ntree=100, cp=0.0001)

RF_class.pred = predict(model_RF_class,  newdata = final_df_test, type= "class")

RF_class_predict <- data.frame(RF_class.pred)
colnames(RF_class_predict)[1] <- "col1"

accuracy_percentage_RF_class <- (sum(RF_class_predict$col1 == final_df_test$stars.x)/nrow(final_df_test))*100
print(paste("Accuracy Percentage:", accuracy_percentage_RF_class, "%"))

matplot(1:100, cbind(RF_class_predict[1:100,], data.frame(final_df_test$stars.x)[1:100,]), type = "l", lty = 1, 
        col = c("red", "blue"), xlab = "First 100 test data points", 
        ylab = "Stars", main = "Classification Random Forest predictions v.s. observed stars")
legend("topright", legend = c("Prediction", "actual"), 
       col = c("red", "blue"), 
       lty = 1)
