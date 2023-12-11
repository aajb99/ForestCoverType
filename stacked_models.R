# Stacking Models ForestCoverType

####### Load Libraries #######
#install.packages('tidyverse')
library(tidyverse)
#install.packages('tidymodels')
library(tidymodels)
#install.packages('DataExplorer')
#install.packages("poissonreg")
# library(poissonreg)
#install.packages("glmnet")
library(glmnet)
#library(patchwork)
# install.packages("rpart")
#install.packages('ranger')
library(ranger)
#install.packages('stacks')
library(stacks)
#install.packages('vroom')
library(vroom)
#install.packages('parsnip')
library(parsnip)
# install.packages('dbarts')
# library(dbarts)
#install.packages('embed')
library(embed)
library(themis)
library(ggplot2)
library(parsnip)
library(bonsai)
library(lightgbm)
library(keras)
# install.packages('baguette')
library(baguette)
# install.packages('stacks')
library(stacks)
library(tune)
##############################

load('RFFinal.RData')
load('BoostFinal.RData')
load('NNFinal.RData')

# Import Dataset:
data_train <- vroom("./data/train.csv") %>%
  mutate(Cover_Type=factor(Cover_Type))# grab training data

data_test <- vroom("./data/test.csv") # grab testing data

# Stacked model

models_stack <- 
  stacks() %>%
  add_candidates(rf_final_mod) %>%
  add_candidates(boost_results1) %>%
  add_candidates(tuned_nn)


# fit stacked model

models_stack <-
  models_stack %>%
  blend_predictions() %>%
  fit_members()


# Prepare preds for kaggle

stack_preds <- 
  stack_mod %>%
  predict(new_data = data_test, type = 'class') %>% # "class" or "prob"
  mutate(Id = data_test$Id) %>%
  mutate(Cover_Type = .pred_class) %>%
  select(Id, Cover_Type)


vroom_write(stack_preds, "./data/stacked_preds_rf_tuned1.csv", delim = ",")





