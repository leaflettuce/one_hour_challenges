library(rattle)
library(ggplot2)
df <-audit

#######
# EDA #
#######
str(df)

# remove unneeded cols
df$ID <- NULL
df$RISK_Adjustment <- NULL
df$TARGET_Adjusted <- NULL
df$Education <- NULL
df$Deductions <- NULL
df$Employment <-  NULL
df$IGNORE_Accounts <- NULL

# quick plots
hist(df$Age)
plot(df$Gender)
plot(df$Marital)

ggplot(aes(x = Education), data  = df) + 
  geom_histogram(stat = "count")

ggplot(aes(x = Occupation), data  = df) + 
  geom_histogram(stat = "count")

ggplot(aes(x = Income), data = df) + 
  geom_histogram()

hist(df$Hours)

ggplot(aes(x = Age, y = Income, color = Gender), data = df) +
  geom_jitter(size = Hours*.05)

############
# Cleaning #
############

# limit outliers
df$Income <- ifelse(df$Income > 150000, 150000, df$Income)

# dummy out factors
sum(is.na(df$Gender)) 
df$Female <- ifelse(df$Gender == "Female", 1, 0)
df$Gender <- NULL

# marriage status
df$Married <- ifelse(df$Marital == "Married" |
                       df$Marital == "Married-spouse-absent", 1, 0)

df$Single <- ifelse(df$Marital == "Divorced" |
                      df$Marital == "Unmarried" |
                      df$Marital == "Widowed", 1, 0)

df$absent <- ifelse(df$Marital == "Absent", 1, 0)

df$Marital <- NULL

# remove na
df = na.omit(df)


#############
# normalize #
#############

min_max_norm <- function(x) {
  return((x - min(x))/(max(x) - min(x)))
}

df[ , c(1,3, 4)] <- lapply(df[ , c(1, 3, 4)], min_max_norm)


occupation <- df$Occupation
df$Occupation <- NULL

str(df)

###########
# Cluster #
###########

# k-means
library(stats)

clusters <- kmeans(df, 3)

# examine (1 - )
clusters$centers
clusters$size

df$Cluster_id <- clusters$cluster

# visualize
ggplot(aes(x = Age, y = Income, color = as.factor(Cluster_id), 
           shape = as.factor(Female)), data = df) + 
          geom_jitter(size = 2)
  
Cluster_ID <- df$Cluster_id
df$Cluster_id <- NULL

# split test -train
train <- df[1:1400, ]
test <- df[1401:1899, ]

# unneeded - tested though
y_train <- occupation[1:1400]
y_test <- occupation[1401:1899]

y_train_cluster <- Cluster_ID[1:1400]
y_test_cluster <- Cluster_ID[1401:1899]

# k-nn predict occupation
library(class)
library(gmodels) 


pred <- knn(train = train, test = test, cl = y_train_cluster, k = 5)

CrossTable(x = y_test_cluster, y = pred, prop.chisq = FALSE)$prop.r

# test to occupation
df$Occupation <- occupation
df$cluster_id <- Cluster_ID

library(plyr)
count(df, c('Occupation','Cluster_ID'))
count(df, c('Cluster_ID','Occupation'))

aggregate(data = df, Age ~ Cluster_ID, mean)
aggregate(data = df, Income ~ Cluster_ID, mean)
