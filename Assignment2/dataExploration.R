#install.packages("corrplot")
library(corrplot)
library(dplyr)

#####################################
# Loading Data 
#####################################
setwd("/Users/sjoerdvisser/Documents/VU/Master/Datamining techniques/Assignment2/Data Mining VU data")

df = read.csv('training_set_VU_DM_2014.csv')
df[df=="NULL"] = NA
for(name in colnames(df[,names(df) != 'date_time'])){
  print(name)
  print(typeof(df[,name]))
  df[,name] = as.numeric(df[,name])
}


#####################################
# Exploration
#####################################
#Count NAs per group
na_count <- colSums(is.na(mysample))
na_count <- data.frame(variable = names(na_count), value = na_count, row.names = NULL)
na_count$value <- sapply(na_count$value, as.numeric)

ggplot(data=na_count, aes(x= reorder(variable, value), y = value)) +
  geom_bar(stat='identity', fill = 'steelblue') +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle('Number of NANs per variable') +
  ylab('Number of NANs') +
  xlab('') +
  theme(plot.title = element_text(hjust = 0.5))

#Plot boxplots per variable
long_data <- melt(mysample)

price_usd <- long_data[long_data$variable=='price_usd']

ggplot(long_data, aes(x=variable, y = value)) + 
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) 

#TODO: create boxplots in smaller groups of variables within similar value ranges


# Number of bookings in comparison to total searches (unique search ids)
bookings <- sum(mysample$booking_bool)
total_searches <- nrow(count(mysample$srch_id))


#function get Summary
#Get statistics about all variables
getSummary <- function(d){
  m = matrix(0, length(colnames(d)), 9)
  rownames(m) =  colnames(d)
  colnames(m) =  c( "Min.","1st Qu.", "Median",  "Mean", "3rd Qu.", "Max.", "Var", "Rows", "NaNs"  )
  i = 1
  for (cat in colnames(d)){
    d2 = d[,cat]
    s = summary(d2)
    m[i, 1:6] =  s[1:6]
    m[i, 7] = var(d2)
    m[i, 8] = length(d2)
    m[i, 9] = sum(is.na(d2))
    i = i+1
  }
  return(round(m, digits = 2))
}

#function getStatsPerCountry
#Get statistics about one variable for each country
getStatsPerCountry = function(d, variable){
  m = matrix(0, length(order(unique(d$prop_country_id))), 7)
  colnames(m) =  c( "Min.","1st Qu.", "Median",  "Mean", "3rd Qu.", "Max.", "Number of rows")
  row.names(m) = unique(d$prop_country_id)
  i = 1
  for (country in unique(d$prop_country_id)){
    d2 = d[d$prop_country_id == country, variable]
    s = summary(d2)
    m[i, 1:6] =  s[1:6]
    m[i, 7] = length(d2)
    i = i+1
  }
  stats = as.data.frame(round(m, digits = 2))
  par(mfrow = c(2,1))
  hist(stats$Mean, breaks = 20, main = paste('Historgram of mean', variable, 'for countries', sep = ' '), xlab = paste('Mean',variable ,sep=' '))
  boxplot(stats$Mean, main = paste('Boxplot of mean', variable, 'for countries', sep = ' '), ylab = paste('Mean',variable ,sep=' '))
  stats = stats[ order(stats$Mean, decreasing = TRUE),]
  return(stats)
}

getSummary(df)
getSummary(df[df$booking_bool==1, ])

#Most values seem plausible. Note the following:
# - The highest displayed price in US dollars is 19.7 million, the highest price booked is 3.7 million dollars.
# - The number of searches where the displayed price in US dollars is higher than 2000 is 30, this accounts for 5072 lines.
x = df[df$price_usd>2000, ]
length(unique(x$site_id))
nrow(x)
df[ df$price_usd<1, ]
# - These prices do not seem plausible
# - We will use the 1.5 interquantile range rule to get rid of these outliers. 
# - The exercise shows us the comp*_rates should be {-1, 0, 1}. 
# - From the summaries we see that these values seem to be shifted to {1, 2, 3}.
# - The comp_rate_percent_diff contains some high values. 



# Correlations

M = cor(df2[,c("srch_adults_count", "srch_children_count", "srch_booking_window", "srch_destination_id", "srch_room_count", "srch_length_of_stay")])
corrplot(M, method="number", type ='upper')
corrplot(M, method="number")

M2 = cor(df2[,7:14], use = 'complete.obs')
corrplot(M2, method = "number", type='upper')
corrplot(M2, method = "circle", type='upper')
