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

# Filter low rated doctors
low_rated <- rmd[rmd$avg_rating < 3, ]

# Filter high rated doctors
high_rated <- rmd[rmd$avg_rating > 3, ]

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
stm_LR <- stm_reviews(low_rated)

 
# # Interpreting the topics by using label() from stm package. 
labelTopics(stm_LR, c(1,2,3,4,5,6,7,8,9,10))

topics <- list(1,2,3,4,5,6,7,8,9,10)
for (t in topics){
  reviews <- findThoughts(stm_LR, texts = out$meta$Reviews,n = 3, topics = t)$docs[[1]]
  plotQuote(reviews, width = 80, text.cex = 0.58, maxwidth = 800, main = paste(c("Topic ", t), collapse = " ")
  )
}

plot(stm_LR, type = "summary", xlim = c(0, 0.3))

out$meta$avg_rating <- as.factor(out$meta$avg_rating)

prep_LR <- estimateEffect(c(1,2,3,4,5,6,7,8,9,10) ~ Gender, stm_LR, meta = out$meta, uncertainty = "Global")
summary(prep_LR, topics = c(1,2,3,4,5,6,7,8,9,10))

#Plot effect of covariates on topics
plot(prep_LR, topics = c(1,2,3,4,5,6,7,8,9,10), covariate = "Gender", cov.value1 = "f", cov.value2 = "m", method = "difference", model = stm_LR, xlab = "Gender", linecol = "blue", xlim = c(-0.5, 0.5), main = "Effect of covariates on topics per gender of doctor")

# Build HLM on topic models for low-rated doctors
topic_probs_low_rated <- as_tibble(stm_LR$theta)

df_lowAll <- as.data.frame(cbind(low_rated, topic_probs_low_rated))

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
low_rated_hlm_topics <- bootstrap_topics(groups, DVs, r, nsim, df_lowAll, low_rated)

###### Building topic models and HLM for high-rated doctors #########

stm_HR <- stm_reviews(high_rated)

# # Interpreting the topics by using label() from stm package. 
labelTopics(stm_HR, c(1,2,3,4,5,6,7,8,9,10))

for (t in topics){
  reviews <- findThoughts(stm_HR, texts = out$meta$Reviews,n = 3, topics = t)$docs[[1]]
  plotQuote(reviews, width = 85, text.cex = 0.57, maxwidth = 900, main = paste(c("Topic ", t), collapse = " ")
  )
}

plot(stm_HR, type = "summary", xlim = c(0, 0.3))

out$meta$avg_rating <- as.factor(out$meta$avg_rating)

prep_HR <- estimateEffect(c(1,2,3,4,5,6,7,8,9,10) ~ Gender, stm_HR, meta = out$meta, uncertainty = "Global")
summary(prep_HR, topics = c(1,2,3,4,5,6,7,8,9,10))

#Plot effect of covariates on topics
plot(prep_HR, topics = c(1,2,3,4,5,6,7,8,9,10), covariate = "Gender", cov.value1 = "f", cov.value2 = "m", method = "difference", model = stm_HR, xlab = "Gender", linecol = "blue", xlim = c(-0.5, 0.5), main = "Effect of covariates on topics per gender of doctor")

# Build HLM on topic models for low-rated doctors
topic_probs_high_rated <- as_tibble(stm_HR$theta)

df_highAll <- as.data.frame(cbind(high_rated, topic_probs_high_rated))

# Build HLM on topic probabilities
high_rated_hlm_topics <- bootstrap_topics(groups, DVs, r, nsim, df_highAll, high_rated)

write.csv(rbind(low_rated_hlm_topics, high_rated_hlm_topics), "D:/Education/HbgUniv/PhD DS/Sex_bias_thesis/SexBias/Online_reviews/Multi-Aspect-Sentiment-Classification-for-Online-Medical-Reviews-main/RateMDs/HLM_STM_Results/STM_HLM_RMD.csv")
