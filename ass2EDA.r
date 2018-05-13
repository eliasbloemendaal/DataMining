library(ggplot2)
library(reshape2)
library(plyr)
library(dplyr)
library(lubridate)


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
#set date_time as datetime format
mysample$date_time <- as.Date(mysample$date_time)

# Apply filters for illogical values

# Filter outliers according to 1.5*IQR rule




  # Jun Wang: Hotel description -> fill missing value with worst case scenario
  #           User data -> highlight matching or mismatching between historical data -> feature engineering
  #           Competitors -> missing values set to 0


#####################################
# Data preparation
#####################################
str(mysample)

# summary per srch_id
by_srch_id <- group_by(mysample, srch_id)
srch_id_sum <- summarize(by_srch_id, min_date = min(date_time),
                         max_date = max(date_time),
                         avg_hist_starrating = mean(visitor_hist_starrating),
                         avg_hist_adr_usd = mean(visitor_hist_adr_usd),
                         
                         nr_searches = n()
                         ) 
# summary per prop_id
by_prop_id <- group_by(mysample, prop_id)

prop_id_sum <- summarize(by_prop_id, avg_prop_starrating = mean(prop_starrating),
                         avg_prop_review_score = mean(prop_review_score),
                         avg_prop_location_score1 = mean(prop_location_score1),
                         avg_prop_location_score2 = mean(prop_location_score2),
                         avg_prop_log_historical_price = mean(prop_log_historical_price),
                         avg_price_usd = mean(price_usd),
                         avg_srch_query_affinity_score = mean(srch_query_affinity_score),
                         std_prop_starrating = sd(prop_starrating),
                         std_prop_review_score = sd(prop_review_score),
                         std_prop_location_score1 = sd(prop_location_score1),
                         std_prop_location_score2 = sd(prop_location_score2),
                         std_prop_log_historical_price = sd(prop_log_historical_price),
                         std_price_usd = sd(price_usd),
                         std_srch_query_affinity_score = sd(srch_query_affinity_score),
                         median_prop_starrating = median(prop_starrating),
                         median_prop_review_score = median(prop_review_score),
                         median_prop_location_score1 = median(prop_location_score1),
                         median_prop_location_score2 = median(prop_location_score2),
                         median_prop_log_historical_price = median(prop_log_historical_price),
                         median_price_usd = median(price_usd),
                         median_srch_query_affinity_score = median(srch_query_affinity_score),
                         click_prob = sum(click_bool)/n(),
                         book_prob = sum(booking_bool)/n()
                          )



#####################################
# Merge data back in
#####################################

mysample <- merge(x = mysample, y = srch_id_sum, by = "srch_id", all.x = TRUE)
mysample <- merge(x = mysample, y = prop_id_sum, by = "prop_id", all.x = TRUE)

####################################
# Date splitting
####################################
mysample$day <- weekdays(mysample$date_time)
mysample$month <- month(mysample$date_time)



#####################################
# Feature engineering
#####################################
# EXP features: categorical features converted into numerical features (Owen Zhang)
get_exp_train <- function(df, col, target){
  #FUNCTION TO BE USED ON TRAIN DATA
  print(col)
  # arrange df on col
  arrange(df, df[col])
  overal_avg <- mean(df[,target])
  
  col_factors <- unique(df[col])
  col_name <- paste(c(col,'exp'),collapse = '_')
  col_exp <- rep(0, nrow(df))
  original_row_nr <- 1
  
  #Loop over all different classes for variable
  for (col_fac in col_factors){
    filtered_df <- filter(df, df[col] == col_fac)

    total_obs <- nrow(filtered_df)
    target_mean <- mean(filtered_df[,target])
    # loop over all rows within filtered df and calculate exp
    for (row in 1:total_obs){
      special_mean = (total_obs * target_mean - filtered_df[row,target]) / (total_obs - 1)
      row_exp = weighted.mean(c(special_mean, target_mean), c(total_obs-1, total_obs))
      col_exp[original_row_nr] <- row_exp
      original_row_nr <- original_row_nr + 1
    }
  }
  # add new column with exp values to df
  df <- cbind(df, col_exp)
  #rename the new column
  names(df)[names(df)=='col_exp'] <- col_name
  return(df)
  
}



get_exp_test <- function(train, test, col, target){
  #Function to get the exp features for a test set, based on the train set
  by_col <- group_by_(train, col)
  train_target_summary <- summarize(by_col, testcolumn = mean(position))
  
  name <- paste(col, 'exp', sep = '_')
  names(train_target_summary)[names(train_target_summary) == "testcolumn"] <- name
  test_set <- merge(x=test, y = train_target_summary, by = col, all.x=TRUE)
  return(test_set)
}
  
target = 'position'
cols_to_get_exp <- c('site_id', 'visitor_location_country_id', 'prop_country_id','prop_id', 'prop_starrating',
                     'prop_review_score', 'srch_destination_id', 'srch_length_of_stay', 'srch_adults_count', 'srch_room_count', 'day', 'month')
#TODO: discuss whether to include ordinal variables in cols_to_get_exp

#mysample <- get_exp_train(mysample, 'day', target)

#Get all expectation features
for (col in cols_to_get_exp){
  mysample <- get_exp_train(mysample, col, target)
}

# TEST SET ONLY

#for (col in cols_to_get_exp){
#  mysample <- get_exp_test(train, test, target)
#}

######################################################################
# Estimated position: position of the same hotel in the same destination in the previous and next search (Owen Zhang)
######################################################################

get_estimated_pos <- function(df, search_line) {
  #Get time difference between searches
  df$time_diff <- as.numeric(difftime(df$date_time, search_line$date_time,unit='days'))
  
  #filter df for location and hotel
  same_prop <- filter(df, df$prop_id == search_line$prop_id)
  filtered <- filter(same_prop, same_prop$srch_destination_id == search_line$srch_destination_id)
  
  
  earlier <- filtered[filtered$time_diff < 0,]
  later <- filtered[filtered$time_diff > 0,]
  
  
  closest_before <- earlier[which.max(earlier$time_diff),]
  closest_after <- later[which.min(later$time_diff),]
  
  result <- mean(c(closest_before$position, closest_after$after))
  
  #drop time_diff coll
  df$time_diff <- NULL
  
  return(result)
}
  

pos_exp <- numeric(nrow(mysample))
for (i in 1:nrow(mysample)){
  pos_exp[i] <- get_estimated_pos(mysample[c('srch_id','date_time','position', 'srch_destination_id','prop_id')], mysample[i,][c('srch_id','date_time','position', 'srch_destination_id','prop_id')])
}

mysample$pos_exp <- pos_exp
# Create features to represent matching or mismatching between historical data and given hotel data
  # E.g. price_diff, stars_diff
mysample <- mutate(mysample, price_diff = abs(avg_hist_adr_usd-price_usd),
                stars_diff = abs(avg_hist_starrating - prop_starrating)
       
       )

# Hotel quality: probability of booking + probability of clicking -> done
# Non-Monoticity of Feature Utility: transform features (Jun Wang) -> done
# Add features for each prop_id: mean, std and median of numeric values --> done




#####################################
# Feature normalization
#####################################
# Normalize hotel and competitor descriptions with respect to different indicators
  # srch_id, prop_id, srch_booking_window, srch_destination_id, prop_country_id

to_normalize <- c('price_usd', 'prop_starrating', 'prop_location_score1', 'prop_location_score1','prop_review_score', 'prop_log_historical_price')
normalize_by <- c('srch_id', 'prop_country_id', 'prop_id')

for (feature in to_normalize){
  for (by in normalize_by){
    
    feature_name <- paste(feature, by, sep='_norm_by_')
    mysample[feature_name] <- ave(mysample[feature], mysample[by], FUN = function(x) x /max(x))
  }
}

#####################################
# Aggregate competitors
#####################################
mysample$comp_rate <- rowSums(mysample[c('comp1_rate','comp2_rate' ,'comp3_rate','comp4_rate','comp5_rate','comp6_rate','comp7_rate','comp8_rate')], na.rm=TRUE)
mysample$comp_inv <- rowSums(mysample[c('comp1_inv','comp2_inv' ,'comp3_inv','comp4_inv','comp5_inv','comp6_inv','comp7_inv','comp8_inv')], na.rm=TRUE)
mysample$comp_rate_percent_diff <- rowMeans(mysample[c('comp1_rate_percent_diff','comp2_rate_percent_diff' ,'comp3_rate_percent_diff','comp4_rate_percent_diff','comp5_rate_percent_diff','comp6_rate_percent_diff','comp7_rate_percent_diff','comp8_rate_percent_diff')], na.rm=TRUE)

#####################################
# Process NAs
#####################################
# Owen Zhang: impute with a negative value
mysample[is.na(mysample)] <- -1


#####################################
# Downsample negative values
#####################################
# Only for train set
#(Owen Zhang) --> rows without booking or click

total_non_booked_clicked <- subset(mysample, click_bool == 0 & booking_bool == 0)
nr_negative <- nrow(total_non_booked_clicked)
ratio_pos <- (nrow(mysample)-nr_negative)/nrow(mysample) # 0.0431

desired_ratio <- 0.1 # 1 : 10
drop_prob <- 1 - (ratio_pos / desired_ratio)

reduced_non_booked_clicked <- sample_frac(total_non_booked_clicked, 1-drop_prob)

train_set <- rbind(reduced_non_booked_clicked, subset(mysample, click_bool == 1 | booking_bool == 1))

#####################################
# Write to csv
#####################################

write.csv(file='DMT_train.csv',x=train_set)

