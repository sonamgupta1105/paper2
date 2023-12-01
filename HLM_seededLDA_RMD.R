library(quanteda)
library(seededlda)
library(readr)
library(BBmisc)
library(dplyr)
library(cld3)
library(Rcpp)

# Read the data in
rm_data_nodups <- read.csv("D:/Education/HbgUniv/PhD DS/Sex_bias_thesis/SexBias/Online_reviews/Multi-Aspect-Sentiment-Classification-for-Online-Medical-Reviews-main/RateMDs/Binary_classification_tm/rateMD_noDups.csv")

# Create a new column with average of P, H, K column scores
rm_data_nodups$avg_rating <- rowMeans(rm_data_nodups[,c('Helpfulness', 'Knowledge')], na.rm=TRUE)

# # Create a subset for obgyn doctors
# 
# ob <- select(filter(rm_data_nodups, rm_data_nodups$Specialty == 'gynecologist-obgyn'), c('Doc_id', 'Doc_Name', 'Specialty','Gender', 'Helpfulness', 'Knowledge', 'Reviews', 'avg_rating'))

low_rating_allspecialty <- rm_data_nodups[rm_data_nodups$avg_rating < 3, ]
#low_rating_allspecialty <- head(low_rating_allspecialty, 100)

seedwords_dict <- dictionary(list(warmth = c('comfortable', 'considerate','interpersonal','nice', 'friendly', 'safe', 'ease', 'approachable', 'sweet', 'pleasant','helpful', 'welcoming', 'accomodating', 'welcomed', 'empathetic', 'compassionate', 'friendly', 'polite', 'lovely', 'courteous', 'cheerful'),
                                  competence = c('superior', 'impressive','competent','arrogant','ambitious','skilled', 'skillful', 'skills', 'exemplary', 'excellence', 'superb', 'stellar', 'capable', 'talented', 'impeccable', 'meticulous', 'proficient', 'condescending', 'abrupt', 'cold', 'rude', 'dismissive', 'impatient', 'unprofessional'),
                                  gendered = c('lady','woman', 'man', 'guy', 'male', 'female', 'she','he', 'her', 'him','gal', 'gentleman', 'fellow'),
                                  appearance = c('attractive','young', 'beautiful','pretty','handsome', 'figure', 'smile', 'smiles', 'aged', 'gorgeous', 'outfit', 'moods')))

# Customized stopwords list used when building document term frequency matrix
stopwordsPL <- readLines("D:/Education/HbgUniv/PhD DS/Sex_bias_thesis/SexBias/Online_reviews/Multi-Aspect-Sentiment-Classification-for-Online-Medical-Reviews-main/RateMDs/stopwords.txt", encoding = "UTF-8", warn = FALSE)

# Topic model for low-rated obgyn doctors
corpus_low_reviews <- corpus(low_rating_allspecialty$Reviews)

# Document frequency matrix
dfmt_all_low <- suppressWarnings(dfm(corpus_low_reviews, remove = stopwordsPL,remove_number = TRUE, remove_punct=TRUE) %>%
                                   dfm_remove(stopwords('english'), min_nchar = 2) %>%
                                   dfm_trim(min_termfreq = 0.5, termfreq_type = "quantile",
                                            max_docfreq = 0.1, docfreq_type = "prop"))
print(dfmt_all_low)
topfeatures(dfmt_all_low, 10)

set.seed(5679) 

# Call the seededlda function and pass the document feature matrix as well as the dictionary of seeded words
# residual = TRUE means output the junk words that are grouped together as the 'other' topic 
slda_all <- textmodel_seededlda(dfmt_all_low, seedwords_dict, beta = 0.5, residual = TRUE)

# get theta values for terms per topic
tidy_topics_all_low <- as_tibble(slda_all$theta, rownames = "Review")

# Print top 20 terms per topic
topic_terms_all <- terms(slda_all, 20)


# Calculate the count of words per topic
#topic_ob_all <- table(topics(slda_all))


df_low_all <- as.data.frame(cbind(low_rating_allspecialty, tidy_topics_all_low))
summary(df_low_all)

# Filter data for high rated doctors
high_rating_all <- rm_data_nodups[rm_data_nodups$avg_rating > 3, ]

# Topic model for high-rated obgyn doctors
all_corpus_high_reviews <- corpus(high_rating_all$Reviews)

# Document frequency matrix
dfmt_all_high <- suppressWarnings(dfm(all_corpus_high_reviews, remove = stopwordsPL,remove_number = TRUE, remove_punct=TRUE) %>%
                                   dfm_remove(stopwords('english'), min_nchar = 2) %>%
                                   dfm_trim(min_termfreq = 0.5, termfreq_type = "quantile",
                                            max_docfreq = 0.1, docfreq_type = "prop"))
#print(dfmt_all_high)
topfeatures(dfmt_all_high, 10)

set.seed(5676) 

# Call the seededlda function and pass the document feature matrix as well as the dictionary of seeded words
# residual = TRUE means output the junk words that are grouped together as the 'other' topic 
slda_all_high <- textmodel_seededlda(dfmt_all_high, seedwords_dict, beta = 0.5, residual = TRUE)

# get theta values for terms per topic
tidy_topics_all_high <- as_tibble(slda_all_high$theta, rownames = "Review")

# Print top 20 terms per topic
topic_terms_all_high <- terms(slda_all_high, 20)
topic_terms_all_high

# Calculate the count of words per topic
topic_all_high <- table(topics(slda_all_high))

# build dataframe with topic probabilities to use for binary classification
df_highAll <- as.data.frame(cbind(high_rating_all, tidy_topics_all_high))

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
                   random = list(~1|Doc_id),
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

groups = c("f", "m") # Independent Variables

# Dependent Variables
DVs = c("appearance", "warmth", "gendered", "competence", "other") #change to column names of tidy_topics_low

r = 1 #Initializes the first row in table1 to save output
nsim = 1000 #Number of samples to test
for(DV in DVs){ #Loops through multiple DVs
  for(group in groups){ #Calculates average means and sds for female and male docs
    data = subset(df_low_all, Gender == group) 
    f = as.formula(paste(DV,'~1', sep='')) #Creates the lm formula to pass to the bootstrap function
    
    bs = bootstrap_values(f,data,nsim) #Calls the bootstrap function to begin
    table1[r, 1] = DV #Save name of the DV being tested in this run
    
    if(group=='f'){ 
      table1[r, 2] = mean(bs$mean) #Calculates and saves average DV mean for female docs
      table1[r, 3] = mean(bs$sd) #Calculates and saves average sd for female docs
      nfemale = length(na.omit(data[[DV]])) #Calculate N for female docs
      table1[r, 9] = quantile(bs$mean, 0.025) ##Calculates and saves lower CI for female docs
      table1[r, 10] = quantile(bs$mean, 0.975) ##Calculates and saves upper CI for female docs
    }
    if(group=='m'){ #Change 1 to label given to Male Docs
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

low_all_hlm_df <- as.data.frame(table1)


# Call bootstrap function for high-rated doctors
r2 = 1 #Initializes the first row in table1 to save output
nsim2 = 1000 #Number of samples to test
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
    data2 = subset(df_highAll, Gender == group) 
    f2 = as.formula(paste(DV,'~1', sep='')) #Creates the lm formula to pass to the bootstrap function
    
    bs2 = bootstrap_values(f2,data2,nsim2) #Calls the bootstrap function to begin
    table2[r2, 1] = DV #Save name of the DV being tested in this run
    
    if(group=='f'){ 
      table2[r2, 2] = mean(bs2$mean) #Calculates and saves average DV mean for female docs
      table2[r2, 3] = mean(bs2$sd) #Calculates and saves average sd for female docs
      nfemale2 = length(na.omit(data2[[DV]])) #Calculate N for female docs
      table2[r2, 9] = quantile(bs2$mean, 0.025) ##Calculates and saves lower CI for female docs
      table2[r2, 10] = quantile(bs2$mean, 0.975) ##Calculates and saves upper CI for female docs
    }
    if(group=='m'){ #Change 1 to label given to Male Docs
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
high_all_hlm_df <- as.data.frame(table2)
write.csv(rbind(low_all_hlm_df, high_all_hlm_df), "D:/Education/HbgUniv/PhD DS/Sex_bias_thesis/SexBias/Online_reviews/Multi-Aspect-Sentiment-Classification-for-Online-Medical-Reviews-main/RateMDs/HLM_seededLDA/HLM_results_seededLDA_RMD.csv")