library(quanteda)
library(readr)
library(BBmisc)
library(dplyr)
library(cld3)
library(Rcpp)
library(MOTE)
library(boot)
library(nlme)
library(tidyverse)
library(tidytext)
library(tools)
library(Hmisc)
library(stm)
library(Rtsne)
library(rsvd)
library(geometry)
library(igraph)

# Function to call and preprocess the file

preprocess_file <- function(rmd_data){
  
  # Create a new column with average of P, H, K column scores
  rmd_data$avg_rating <- rowMeans(rmd_data[,c('Helpfulness', 'Knowledge')], na.rm=TRUE)
  # Excluding empty strings
  rmd_data <- rmd_data[complete.cases(rmd_data),] 
}

stopwordsPL <- readLines("D:/Education/HbgUniv/PhD DS/Sex_bias_thesis/SexBias/Online_reviews/Multi-Aspect-Sentiment-Classification-for-Online-Medical-Reviews-main/RateMDs/stopwords.txt", encoding = "UTF-8", warn = FALSE)
# Read the dataset downloaded from GitHub Repo as referenced in the main paper
rmd_data <- read.csv("rateMD_noDups.csv")
rmd <- preprocess_file(rmd_data)


# Function to call STM on low-rated doctors

stm_reviews <- function(reviews_data){
  processed <- textProcessor(reviews_data$Reviews, metadata = reviews_data, customstopwords = stopwordsPL)
  
  # To find the right threshold for number of words, tokens and documents to use for the analysis
  plotRemoved(processed$documents, lower.thresh = seq(1, 75, by = 25))
  out <<- prepDocuments(processed$documents, processed$vocab, processed$meta, lower.thresh = 0)
  
  topics_reviews <- stm(documents = out$documents, vocab = out$vocab, K = 10, prevalence = ~ out$meta$Gender ,data = out$meta, init.type = "Spectral")
  
  return(topics_reviews)
  
}

# Build topic models on low-rated doctors
stm_all <- stm_reviews(rmd)


# # Interpreting the topics by using label() from stm package. 
labelTopics(stm_all, c(1,2,3,4,5,6,7,8,9,10))

topics <- list(1,2,3,4,5,6,7,8,9,10)
for (t in topics){
  reviews <- findThoughts(stm_all, texts = out$meta$Reviews,n = 3, topics = t)$docs[[1]]
  plotQuote(reviews, width = 80, text.cex = 0.58, maxwidth = 800, main = paste(c("Topic ", t), collapse = " ")
  )
}

plot(stm_all, type = "summary", xlim = c(0, 0.6), n = 10, text.cex = 0.94)

out$meta$avg_rating <- as.factor(out$meta$avg_rating)

prep_gender <- estimateEffect(c(1,2,3,4,5,6,7,8,9,10) ~ Gender, stm_all, meta = out$meta, uncertainty = "Global")
summary(prep_gender, topics = c(1,2,3,4,5,6,7,8,9,10))

#Plot effect of gender on topics
plot(prep_gender, topics = c(1,2,3,4,5,6,7,8,9,10), covariate = "Gender", cov.value1 = "f", cov.value2 = "m", method = "difference", model = stm_all, xlab = "Reviews of Female Docs...Male Docs", linecol = "blue", xlim = c(-0.25, 0.25), main = "Effect of covariates on topics per gender of doctor", labeltype = 'custom',
     custom.labels = c('Detailed Results', 'Positive Overall Experience', 'Good Communication',
                       'Healthy Prescriptions', 'Unprofessional Appointment Exp.', 'Interpersonal & Knowledgable',
                       'Decent Patient Exp.', 'Bedside Manner', 'Caring & Personable Physicians',
                       'Technical Competency'))

# 'Unprofessional Appointment Experience', 'Positive Hospital Exp.', 'Positive Overall Exp.',
#'Friendly Office Exp.',
# 'Bedside Manner', 'Good Communication', 'Positive Traits', 'Decent Patient Exp.',
# 'Caring Physicians', 'Interpersonal & Knowledgable'))


# prep_rating <- estimateEffect(c(1,2,3,4,5,6,7,8,9,10) ~ avg_rating, stm_all, meta = out$meta, uncertainty = "Global")
# summary(prep_rating, topics = c(1,2,3,4,5,6,7,8,9,10))
# 
# #Plot effect of gender on topics
# plot(prep_rating, topics = c(1,2,3,4,5,6,7,8,9,10), covariate = "avg_rating", cov.value1 = 1, cov.value2 = 5, method = "difference", model = stm_all, xlab = "Reviews of Female Docs...Male Docs", linecol = "blue", xlim = c(-0.5, 0.5), main = "Effect of covariates on topics per avg rating for doctor", labeltype = 'custom',
#      custom.labels = c('Detailed Results', 'Good Treatment Experience', 'Clear Communication',
#                        'Healthy Prescriptions', 'Staff Inconsistencies', 'Knowledgable Physicians',
#                        'Doctor Recommendations', 'Good Bedside Manner', 'Caring & Personable Physicians',
#                        'Technical Competency'))


# Build HLM on topic models for low-rated doctors
topic_probs_all <- as_tibble(stm_all$theta)

df_All <- as.data.frame(cbind(rmd, topic_probs_all))

####make the table####
# This just creates an empty table to save the means, sds, and effect sizes with CIs
table1 = matrix(NA, ncol = 12, nrow = 10) #Change nrow to number of DV considered
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
    
    #create a dataset
    d <- dataset[sample(1:nrow(dataset), 
                        size = nrow(dataset),
                        replace = TRUE), ]
    #test the model
    tryCatch({
      
      model1 = lme(formula, 
                   data = d, 
                   method = "ML", 
                   na.action = "na.omit",
                   random = list(~1|Doc_Name),
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
####Run Bootstrap Function on Data####
bootstrap_topics <- function(groups, DVs, r, nsim, data_df,rated_df){
  for(DV in DVs){ #Loops through multiple DVs
    print(DV)
    for(group in groups){ #Calculates average means and sds for female and male docs
      print(group)
      data = subset(data_df, rated_df$Gender==group) #Change dataset to df name, change Gender to column name with Gender
      f = as.formula(paste(DV,'~1', sep='')) #Creates the lm formula to pass to the bootstrap function
      bs = bootstrap_values(f,data,nsim) #Calls the bootstrap function to begin
      table1[r, 1] = DV #Save name of the DV being tested in this run
      if(group=='f'){ #Change 0 to label given to Female Docs
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
                    n1 = nfemale, 
                    n2 = nmale, 
                    a = .05)
    table1[r, 6] = mdiff$d
    table1[r, 7] = mdiff$dlow
    table1[r, 8] = mdiff$dhigh
    r = r + 1 #Moves to the next row in table1 to save the next set of results
  }
  hlm_df <- as.data.frame(table1)
  return(hlm_df)
}

groups = c("f", "m") # change to whatever labels you gave gender
DVs = c("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10") 
r = 1 #Initializes the first row in table1 to save output
nsim = 1000 #Number of samples to test
all_hlm_topics <- bootstrap_topics(groups, DVs, r, nsim, df_All, rmd)
