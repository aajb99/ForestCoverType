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
##############################

# Import Dataset:
data_train <- vroom("./data/train.csv") %>%
  mutate(Cover_Type=factor(Cover_Type))# grab training data

# data_train

################################################################################
# Recipe/Bake

rFormula <- Cover_Type ~ .

fct_lgbm_recipe <- recipe(Cover_Type ~ ., data = data_train) %>%
  update_role(Id, new_role = "Id") %>%
  step_mutate(Id = factor(Id)) %>%
  step_mutate_at(all_outcomes(), fn = factor, skip = TRUE)

prepped_recipe_boost <- prep(fct_lgbm_recipe) # preprocessing new data
baked_data_boost <- bake(prepped_recipe_boost, new_data = data_train)


################################################################################
####################
##### Boosting #####
####################

boost_model <- boost_tree(trees = 100,
                          tree_depth = 5,
                          learn_rate = .05,
                          mtry = 15,
                          min_n = 12,
                          loss_reduction = 0
                          ) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

boost_wf <- workflow() %>%
  add_recipe(fct_lgbm_recipe) %>%
  add_model(boost_model)

# Split data for CV
folds <- vfold_cv(data_train, v = 5, repeats = 1)

# Run CV
# tuned_boost <- boost_wf %>%
#   tune_grid(resamples = folds,
#             grid = boost_tuneGrid,
#             metrics = metric_set(accuracy))
# 
# bestTune <- tuned_boost %>%
#   select_best('accuracy')

final_wf <- boost_wf %>%
  fit(data = data_train)

data_test <- vroom("./data/test.csv") # grab testing data

## CV tune, finalize and predict here and save results22
## This takes a few min (10 on my laptop) so run it on becker if you want
# Kaggle DF
lgbm_predictions_boost <- predict(final_wf,
                                 new_data=data_test,
                                 type="class") %>% # "class" or "prob"
  mutate(Id = data_test$Id) %>%
  #mutate(ACTION = ifelse(.pred_1 > .95, 1, 0)) %>%
  mutate(Cover_Type = .pred_class) %>%
  select(Id, Cover_Type)

vroom_write(lgbm_predictions_boost, "./data/lgbm_pred_boost.csv", delim = ",")
# save(file = 'ggg_nn_wf.RData', list = c('final_wf'))
# load('ggg_nn_wf.RData')






