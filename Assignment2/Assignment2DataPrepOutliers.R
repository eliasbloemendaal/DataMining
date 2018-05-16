#install.packages("corrplot")
library(corrplot)
library(dplyr)

####################################################################
##################### LOADING DATA #################################
####################################################################
setwd("/Users/sjoerdvisser/Documents/VU/Master/Datamining techniques/Assignment2/Data Mining VU data")

df = read.csv('training_set_VU_DM_2014.csv')
df[df=="NULL"] = NA
for(name in colnames(df[,names(df) != 'date_time'])){
  print(name)
  print(typeof(df[,name]))
  df[,name] = as.numeric(df[,name])
}

##############################################################################
##################### STATISTICS  & OUTLIERS #################################
##############################################################################

findOutliers = function(x, variable, factor){
  lowerq = quantile(x[,variable])[2]
  upperq = quantile(x[,variable])[4]
  iqr = upperq - lowerq 
  upperTreshold = (iqr * factor) + upperq
  lowerTreshold = lowerq - (iqr * factor)
  result <- x[x[,variable]>upperTreshold | x[,variable]<lowerTreshold, ]
  print(lowerTreshold[1])
  print(upperTreshold[1])
  return(result)
}

removeOutlierSearches = function(x, variable, factor){
  outliers = findOutliers(x, variable, factor)
  print(nrow(outliers))
  outlierSearchId = unique(outliers$srch_id)
  x = x[!(x$srch_id %in% outlierSearchId), ]
  return(x)
}

removeOutlierRows = function(x, variable, factor){
  x$id = rownames(x)
  outliers = findOutliers(x, variable, factor)
  x = x[!(x$id %in% outliers$id),]
  x$id = NULL
  return(x)
}

setOutliersNa = function(x, variable, factor){
  x$id = rownames(x)
  outliers = findOutliers(x, variable, factor)
  x[(x$id %in% outliers$id), variable] = NA
  x$id = NULL
  return(x)
}

setOutliersValue = function(x, variable, factor, value){
  x$id = rownames(x)
  outliers = findOutliers(x, variable, factor)
  x[(x$id %in% outliers$id), variable] = value
  x$id = NULL
  return(x)
}

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

getStatsPerCountry = function(d, variable){
  m = matrix(0, length(order(unique(d$prop_country_id))), 7)
  colnames(m) =  c( "Min.","1st Qu.", "Median",  "Mean", "3rd Qu.", "Max.", "Number of rows")
  row.names(m) = unique(d$prop_coutry_id)
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
# - We will use the 1.5 interquantile range rule to get rid of these outliers. 

# - The exercise shows us the comp*_rates should be {-1, 0, 1}. 
# - From the summaries we see that these values seem to be shifted to {1, 2, 3}.

# - The comp_rate_percent_diff contains some high values.  


#price_usd
##############################################################################
##running these plots takes a lot of time, plots are interesting for report. 
par(mfrow = c(2,2))
# plot(df$price_usd)
# scatter.smooth(df$price_usd)
# hist(df$price_usd)
# boxplot(df$price_usd)

#This function shows that some countries have a particularly high average for price_usd and a high max price_usd.
stats = getStatsPerCountry(df, 'price_usd')

nrow(df)  #4958347
df1 = removeOutlierSearches(df, 'price_usd', 1.5)
nrow(df1)  #3084873
df2 = removeOutlierRows(df, 'price_usd', 1.5)
nrow(df2) 
df3 = setOutliersNa(df, 'price_usd', 1.5)
nrow(df3)
df4 = setOutliersValue(df, 'price_usd', 1.5, -1)


#comp1_ratepercent_diff
##############################################################################
par(mfrow = c(2,2))
# plot(df$comp1_ratepercent_diff)
# scatter.smooth(df$comp1_ratepercent_diff)
# hist(df$comp1_ratepercent_diff)
# boxplot(df$comp1_ratepercent_diff)

nrow(df)  
df1 = removeOutlierSearches(df, 'comp1_ratepercent_diff', 1.5)
nrow(df1)  
df2 = removeOutlierRows(df, 'comp1_ratepercent_diff', 1.5)
nrow(df2)
df3 = setOutliersNa(df, 'comp1_ratepercent_diff', 1.5)

#comp2_ratepercent_diff
##############################################################################
par(mfrow = c(2,2))
# plot(df$comp2_ratepercent_diff)
# scatter.smooth(df$comp2_ratepercent_diff)
# hist(df$comp2_ratepercent_diff)
# boxplot(df$comp2_ratepercent_diff)

nrow(df)  
df1 = removeOutlierSearches(df, 'comp2_ratepercent_diff', 1.5)
nrow(df1)  
df2 = removeOutlierRows(df, 'comp2_ratepercent_diff', 1.5)
nrow(df2)
df3 = setOutliersNa(df, 'comp2_ratepercent_diff', 1.5)

#comp3_ratepercent_diff
##############################################################################



#etc. 



####################################################################
##################### CORRELATIONS #################################
####################################################################


M = cor(df2[,c("srch_adults_count", "srch_children_count", "srch_booking_window", "srch_destination_id", "srch_room_count", "srch_length_of_stay")])
corrplot(M, method="number", type ='upper')
corrplot(M, method="number")

M2 = cor(df2[,7:14], use = 'complete.obs')
corrplot(M2, method = "number", type='upper')
corrplot(M2, method = "circle", type='upper')




# [1] "srch_id"  search id and date time moeten gelijk zijn 
# [1] "date_time"
# [1] "site_id"   
# [1] "visitor_location_country_id"
# [1] "visitor_hist_starrating"
# [1] "visitor_hist_adr_usd"
# [1] "prop_country_id"
# [1] "prop_id"
# [1] "prop_starrating"
# [1] "prop_review_score"
# [1] "prop_brand_bool"
# [1] "prop_location_score1"
# [1] "prop_location_score2"
# [1] "prop_log_historical_price"
# [1] "position"
# [1] "price_usd"                 "waarom zijn er uitschieters naarboven? Welke landen geven die hoge prijs? hoeveel datapunten zijn er met deze prijs?"
# [1] "promotion_flag"
# [1] "srch_destination_id"
# [1] "srch_length_of_stay"
# [1] "srch_booking_window"
# [1] "srch_adults_count"
# [1] "srch_children_count"
# [1] "srch_room_count"
# [1] "srch_saturday_night_bool"
# [1] "srch_query_affinity_score"
# [1] "orig_destination_distance"
# [1] "random_bool"
# [1] "comp1_rate"
# [1] "comp1_inv"
# [1] "comp1_rate_percent_diff"
# [1] "comp2_rate"
# [1] "comp2_inv"
# [1] "comp2_rate_percent_diff"
# [1] "comp3_rate"
# [1] "comp3_inv"
# [1] "comp3_rate_percent_diff"
# [1] "comp4_rate"
# [1] "comp4_inv"
# [1] "comp4_rate_percent_diff"
# [1] "comp5_rate"
# [1] "comp5_inv"
# [1] "comp5_rate_percent_diff"
# [1] "comp6_rate"
# [1] "comp6_inv"
# [1] "comp6_rate_percent_diff"
# [1] "comp7_rate"
# [1] "comp7_inv"
# [1] "comp7_rate_percent_diff"
# [1] "comp8_rate"
# [1] "comp8_inv"
# [1] "comp8_rate_percent_diff"
# [1] "click_bool"
# [1] "gross_bookings_usd"
# [1] "booking_bool"


