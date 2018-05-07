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
#set date_time as datetime format
mysample$date_time <- as.Date(mysample$date_time)

# Apply filters for illogical values

# Filter outliers according to 1.5*IQR rule



# Process NAs
  # Owen Zhang: impute with a negative value -> after all other cleaning steps to prevent interference
mysample[is.na(mysample)] <- -1

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


# Group by srch_destination_id and compare quality of hotels
by_srch_des_id <- group_by(mysample, srch_destination_id)

#srch_des_sum <- summarize(by_srch_des_id, )

#####################################
# Merge data back in
#####################################

mysample <- merge(x = mysample, y = srch_id_sum, by = "srch_id", all.x = TRUE)
mysample <- merge(x = mysample, y = prop_id_sum, by = "prop_id", all.x = TRUE)



#####################################
# Feature engineering
#####################################
# EXP features: categorical features converted into numerical features (Owen Zhang)
get_exp_train <- function(df, col, target){
  #FUNCTION TO BE USED ON TRAIN DATA
  
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
      special_mean = (total_obs * target_mean - filtered_df[row,col]) / (total_obs - 1)
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

get_exp_test <- function(train, test_set, col, target){
  #Function to get the exp features for a test set, based on the train set
  # TO BE TESTED
  by_col <- group_by(train, col)
  train_target_summary <- summarize(by_col, paste(c(col,'exp'), collapse = '_') = mean(target))
  test_set <- merge(x=test_set, y = train_target_summary, by = col, all.x=TRUE)
  return(test_set)
  }
  
target = 'position'
cols_to_get_exp <- c('site_id', 'visitor_location_country_id', 'prop_country_id','prop_id', 'prop_starrating',
                     'prop_review_score', 'srch_destination_id', 'srch_length_of_stay', 'srch_adults_count', 'srch_room_count')
#TODO: discuss whether to include ordinal variables in cols_to_get_exp

#Get all expectation features
for (col in cols_to_get_exp){
  mysample <- get_exp_train(mysample, col, target)
}

# TEST
my_sample <- get_exp_test(mysample, my_sample, 'site_id', 'position')


# Estimated position: position of the same hotel in the same destination in the previous and next search (Owen Zhang)

get_estimated_pos <- function(df, search_line) {
  #Get time difference between searches
  df$time_diff <- as.numeric(df$date_time - search_line$date_time)
  
  #filter df for location and hotel
  df <- df[(df$prop_id == search_line$prop_id) && (df$srch_destination_id == search_line$srch_destination_id)]
  earlier = df[df['time_diff'] < 0,]
  later <- df[df['time_diff'] > 0,]
  
  closest_before = df[which(earlier['time_diff'] == max(earlier['time_diff'])),]
  closest_after = df[which(later['time_diff'] == min(later['time_diff'])),]
  
  closest = c(closest_before, closest_after)
  result <- mean(closest$position)
  return(result)
}
  
#Get expected position for each srch
mysample$pos_exp1 <- apply(mysample,1,function(x) get_estimated_pos(mysample, x))


pos_exp <- numeric(nrow(mysample))
for (i in 1:nrow(mysample)){
  #print(is.recursive(mysample[i,]))
  #print(mysample[i,])
  print(i)
  print(mysample[i,][c('date_time', 'srch_id', 'srch_destination_id')])
  pos_exp[i] <- get_estimated_pos(mysample, mysample[i,])
}

mysample$pos_exp <- pos_exp
# Create features to represent matching or mismatching between historical data and given hotel data
  # E.g. price_diff, stars_diff
mutate(mysample, price_diff = abs(avg_hist_adr_usd.y-price_usd),
                stars_diff = abs(avg_hist_starrating.y - prop_starrating)
       
       )

# Hotel quality: probability of booking + probability of clicking -> done
# Non-Monoticity of Feature Utility: transform features (Jun Wang)
# Add features for each prop_id: mean, std and median of numeric values --> done




#####################################
# Feature normalization
#####################################
# Normalize hotel and competitor descriptions with respect to different indicators
  # srch_id, prop_id, srch_booking_window, srch_destination_id, prop_country_id

####################################
# Downsample negative values
####################################
# Only for train set
#(Owen Zhang) --> rows without booking or click
