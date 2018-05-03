library(ggplot2)
library(reshape2)
library(plyr)
library(dplyr)


#data <- read.csv(file='C:\\Users\nvanderheijden\\Desktop\\Data Mining Techniques\\Assignment 2\\Data Mining VU data\\Data Mining VU datatraining_set_VU_DM_2014.csv', header = TRUE, sep=',')
mysample <- training_set_VU_DM_2014[sample(1:nrow(training_set_VU_DM_2014), 10000,replace=FALSE),]
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



#####################################
# Data manipulation
#####################################
# Start with downsampled subset
mysample

# Apply filters for illogical values

# Filter outliers according to 1.5*IQR rule

# Downsample negative values (Owen Zhang) --> rows without booking or click


# Process NAs
  # Owen Zhang: impute with a negative value -> after all other cleaning steps to prevent interference
  # Jun Wang: Hotel description -> fill missing value with worst case scenario
  #           User data -> highlight matching or mismatching between historical data
  #           Competitors -> missing values set to 0







#####################################
# Feature engineering
#####################################
# EXP features: categorical features converted into numerical features (Owen Zhang)
# Estimated position: position of the same hotel in the same destination in the previous and next search (Owen Zhang)
# Create features to represent matching or mismatching between historical data and given hotel data
  # E.g. price_diff, stars_diff
# Hotel quality: probability of booking + probability of clicking
# Non-Monoticity of Feature Utility: transform features (Jun Wang)
# Add features for each prop_id: mean, std and median of numeric values

#####################################
# Feature normalization
#####################################
# Normalize hotel and competitor descriptions with respect to different indicators
  # srch_id, prop_id, srch_booking_window, srch_destination_id, prop_country_id

