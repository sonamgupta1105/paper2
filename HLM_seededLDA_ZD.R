library(quanteda)
library(seededlda)
library(readr)
library(BBmisc)
library(dplyr)
library(cld3)
library(Rcpp)
library(MOTE)
library(boot)
library(nlme)
library(MOTE)
library(tools)
library(Hmisc)

# Read the dataset downloaded from GitHub Repo as referenced in the main paper
zdata <- read.csv("D:/Education/HbgUniv/PhD DS/Sex_bias_thesis/SexBias/Online_reviews/MLReviewsToRatings/DATA/SeededLDA_LDA_binClass/zocdoc_data.csv")

# Rename column names
colnames(zdata) <- c("Doc_name", "Gender", "Reviews", "OverallRating", "BedsideMannerRating", "WaitTimeRating")

# Remove duplicates
zdata_nodups <- subset(zdata, !duplicated(subset(zdata, select=c(Reviews))))

# Remove non-English reviews from the main non-duplicated dataframe
zdata_nodups <- zdata_nodups[!(detect_language(zdata_nodups$Reviews) != 'en'), ]

# Excluding empty strings
zdata_nodups <- zdata_nodups[complete.cases(zdata_nodups),]


# Create a new column with average of BedsideManner and WaitTime column scores
zdata_nodups$avg_rating <- rowMeans(zdata_nodups[,c('BedsideMannerRating', 'WaitTimeRating')], na.rm=TRUE)

seedwords_dict <- dictionary(list(warmth = c('comfortable', 'considerate','interpersonal','nice', 'friendly', 'welcomed', 'uncomfortable', 'welcome', 'ease', 'respectful', 'reassuring', 'sympathetic', 'gracious', 'polite', 'welcoming', 'pleasant', 'accomodating', 'lovely', 'helpful', 'professional', 'sweet'),
                                  competence = c('superior', 'impressive','competent','arrogant','ambitious','skilled', 'skillful','skills', 'consistently', 'reasonable', 'talented', 'intelligent', 'professional', 'smart', 'capable', 'insulting'),
                                  gendered = c('lady','woman', 'man', 'guy', 'male', 'female', 'she','he', 'her', 'him', 'ladies', 'girls', 'stylish'),
                                  appearance = c('attractive','young', 'beautiful','pretty','handsome', 'figure', 'beautifully', 'charming', 'soft')))

# Customized stopwords list used when building document term frequency matrix

stopwordsPL <- readLines("D:/Education/HbgUniv/PhD DS/Sex_bias_thesis/SexBias/Online_reviews/Multi-Aspect-Sentiment-Classification-for-Online-Medical-Reviews-main/RateMDs/stopwords.txt", encoding = "UTF-8", warn = FALSE)

# Filter low rated doctors
low_rated <- zdata_nodups[zdata_nodups$avg_rating < 3, ]

# Topic model for low-rated primary care doctors
zd_low_reviews <- corpus(low_rated$Reviews)

# Document frequency matrix
dfmt_low <- suppressWarnings(dfm(zd_low_reviews, remove = stopwordsPL,remove_number = TRUE, remove_punct=TRUE) %>%
                               dfm_remove(stopwords('english'), min_nchar = 2) %>%
                               dfm_trim(min_termfreq = 0.5, termfreq_type = "quantile",
                                        max_docfreq = 0.1, docfreq_type = "prop"))
#print(dfmt_low)
topfeatures(dfmt_low, 10)

set.seed(5675) 

# Call the seededlda function and pass the document feature matrix as well as the dictionary of seeded words
# residual = TRUE means output the junk words that are grouped together as the 'other' topic 
slda_low <- textmodel_seededlda(dfmt_low, seedwords_dict, beta = 0.5, residual = TRUE)

# get theta values for terms per topic
tidy_topics_low <- as_tibble(slda_low$theta, rownames = "Reviews")

# Print top 20 terms per topic
topic_terms_low <- terms(slda_low, 20)
print('Top 20 words per topic for low-rated PCP: \n')
print(as.table(topic_terms_low))

# Calculate the count of words per topic
topic_low <- table(topics(slda_low))

# Combine the main data and theta values from topic modeling
tidy_topics_low$Reviews <- NULL
df_low <- as.data.frame(cbind(low_rated, tidy_topics_low))
summary(df_low)

# Analysis for high-rated doctors:

# Filter high rated doctors
high_rated <- zdata_nodups[zdata_nodups$avg_rating >= 3, ]

# Topic model for high-rated primary care doctors
zd_high_reviews <- corpus(high_rated$Reviews)

# Document frequency matrix
dfmt_high <- suppressWarnings(dfm(zd_high_reviews, remove = stopwordsPL,remove_number = TRUE, remove_punct=TRUE) %>%
                               dfm_remove(stopwords('english'), min_nchar = 2) %>%
                               dfm_trim(min_termfreq = 0.5, termfreq_type = "quantile",
                                        max_docfreq = 0.1, docfreq_type = "prop"))
#print(dfmt_high)
topfeatures(dfmt_high, 10)

set.seed(5676) 

# Call the seededlda function and pass the document feature matrix as well as the dictionary of seeded words
# residual = TRUE means output the junk words that are grouped together as the 'other' topic 
slda_high <- textmodel_seededlda(dfmt_high, seedwords_dict, beta = 0.5, residual = TRUE)

# get theta values for terms per topic
tidy_topics_high <- as_tibble(slda_high$theta, rownames = "Reviews")

# Print top 20 terms per topic
topic_terms_high <- terms(slda_high, 20)
print('Top 20 words per topic for high-rated PCP: \n')
print(topic_terms_high)

# Calculate the count of words per topic
topic_high <- table(topics(slda_high))

# Combine the main data and theta values from topic modeling
tidy_topics_high$Reviews <- NULL
df_high <- as.data.frame(cbind(high_rated, tidy_topics_high))



########### HLM Analysis ################

#### make the table ####
# This just creates an empty table to save the means, sds, and effect sizes with CIs
table1 = matrix(NA, ncol = 12, nrow = 1) 
colnames(table1) = c("DV", "M_female", 
                     "SD_female", "M_male", "SD_male", 
                     "d", "dlow", "dhigh", "bcilowF", "bcihighF", 
                     "bcilowM", "bcihighM")
table1 = as.data.frame(table1)

####bootstrap function####
#Repeated samples data to run HLM with random intercepts for DocName and returns means and standard deviations for each sample
bootstrap_values <- function(formula, dataset, nsim){
  
  store_mean <- rep(NA, nsim)
  store_sd <- rep(NA, nsim)  
  attempts <- 0
  #loop until you have enough
  while(attempts < nsim){
    
    d <- dataset[sample(1:nrow(dataset), 
                        size = nrow(dataset),
                        replace = TRUE), ]
    #test the model
    tryCatch({
      
      model1 = lme(formula, 
                   data = d, 
                   method = "ML", 
                   na.action = "na.omit",
                   random = list(~1|Doc_name),
                   control=lmeControl(opt = "nlminb")) 
      meanvalue = summary(model1)$tTable[1]
      sdvalue = summary(model1)$tTable[2] * sqrt(nrow(d))
      attempts <- attempts + 1
      store_mean[attempts] <- meanvalue
      store_sd[attempts] <- sdvalue
      return(store_mean, store_sd, attempts)
    }, error = function(x){})
  }
  
  return(list("mean" = store_mean, "sd" = store_sd))
}

# Call bootstrap function for low-rated doctors

groups = c("Female", "Male") # Independent Variables

# Dependent Variables
DVs = c("appearance", "warmth", "gendered", "competence", "other") #change to column names of tidy_topics_low

r = 1 #Initializes the first row in table1 to save output
nsim = 1000 #Number of samples to test
for(DV in DVs){ #Loops through multiple DVs
  for(group in groups){ #Calculates average means and sds for female and male docs
    data = subset(df_low, Gender == group) 
    f = as.formula(paste(DV,'~1', sep='')) #Creates the lm formula to pass to the bootstrap function
    
    bs = bootstrap_values(f,data,nsim) #Calls the bootstrap function to begin
    table1[r, 1] = DV #Save name of the DV being tested in this run
    
    if(group=='Female'){ 
      table1[r, 2] = mean(bs$mean) #Calculates and saves average DV mean for female docs
      table1[r, 3] = mean(bs$sd) #Calculates and saves average sd for female docs
      nfemale = length(na.omit(data[[DV]])) #Calculate N for female docs
      table1[r, 9] = quantile(bs$mean, 0.025) ##Calculates and saves lower CI for female docs
      table1[r, 10] = quantile(bs$mean, 0.975) ##Calculates and saves upper CI for female docs
    }
    if(group=='Male'){ #Change 1 to label given to Male Docs
      table1[r, 4] = mean(bs$mean) #Calculates and saves average DV mean for male docs
      table1[r, 5] = mean(bs$sd) #Calculates and saves average sd for male docs
      nmale = length(na.omit(data[[DV]])) #Calculate N for male docs
      table1[r, 11] = quantile(bs$mean, 0.025) ##Calculates and saves lower CI for male docs
      table1[r, 12] = quantile(bs$mean, 0.975) ##Calculates and saves upper CI for male docs
    }
  }
  #Calculates and saves Cohen's d and its CI between male and female docs
  mdiff = d.ind.t(m1 = table1[r,2], m2 = table1[r,4],
                  sd1 = table1[r,3], sd2 = table1[r,5],
                  n1 = as.numeric(nfemale),
                  n2 = as.numeric(nmale),
                  a = 0.05)
  table1[r, 6] = mdiff$d
  table1[r, 7] = mdiff$dlow
  table1[r, 8] = mdiff$dhigh
  r = r + 1 #Moves to the next row in table1 to save the next set of results
}

# Call bootstrap function for high-rated doctors
r2 = 1 #Initializes the first row in table1 to save output
nsim2 = 1000  #Number of samples to test
table2 = matrix(NA, ncol = 12, nrow = 1) 
colnames(table2) = c("DV", "M_female", 
                     "SD_female", "M_male", "SD_male", 
                     "d", "dlow", "dhigh", "bcilowF", "bcihighF", 
                     "bcilowM", "bcihighM")
table2 = as.data.frame(table2)

for(DV in DVs){ #Loops through multiple DVs
  #print(DV)
  for(group in groups){ #Calculates average means and sds for female and male docs
    #print(group)
    data2 = subset(df_high, Gender == group) 
    f2 = as.formula(paste(DV,'~1', sep='')) #Creates the lm formula to pass to the bootstrap function
    
    bs2 = bootstrap_values(f2,data2,nsim2) #Calls the bootstrap function to begin
    table2[r2, 1] = DV #Save name of the DV being tested in this run
    
    if(group=='Female'){ 
      table2[r2, 2] = mean(bs2$mean) #Calculates and saves average DV mean for female docs
      table2[r2, 3] = mean(bs2$sd) #Calculates and saves average sd for female docs
      nfemale2 = length(na.omit(data2[[DV]])) #Calculate N for female docs
      table2[r2, 9] = quantile(bs2$mean, 0.025) ##Calculates and saves lower CI for female docs
      table2[r2, 10] = quantile(bs2$mean, 0.975) ##Calculates and saves upper CI for female docs
    }
    if(group=='Male'){ #Change 1 to label given to Male Docs
      table2[r2, 4] = mean(bs2$mean) #Calculates and saves average DV mean for male docs
      table2[r2, 5] = mean(bs2$sd) #Calculates and saves average sd for male docs
      nmale2 = length(na.omit(data2[[DV]])) #Calculate N for male docs
      table2[r2, 11] = quantile(bs2$mean, 0.025) ##Calculates and saves lower CI for male docs
      table2[r2, 12] = quantile(bs2$mean, 0.975) ##Calculates and saves upper CI for male docs
    }
  }
  #Calculates and saves Cohen's d and its CI between male and female docs
  mdiff2 = d.ind.t(m1 = table2[r2,2], m2 = table2[r2,4],
                  sd1 = table2[r2,3], sd2 = table2[r2,5],
                  n1 = as.numeric(nfemale2),
                  n2 = as.numeric(nmale2),
                  a = 0.05)
  table2[r2, 6] = mdiff2$d
  table2[r2, 7] = mdiff2$dlow
  table2[r2, 8] = mdiff2$dhigh
  r2 = r2 + 1 #Moves to the next row in table2 to save the next set of results
}
low_all_hlm_df <- as.data.frame(table1)
high_all_hlm_df <- as.data.frame(table2)
write.csv(rbind(low_all_hlm_df, high_all_hlm_df), "D:/Education/HbgUniv/PhD DS/Sex_bias_thesis/SexBias/Online_reviews/MLReviewsToRatings/DATA/SeededLDA_LDA_binClass/HLM_results_seededLDA_ZD.csv")
