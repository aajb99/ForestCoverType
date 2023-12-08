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
install.packages('baguette')
library(baguette)
install.packages('stacks')
library(stacks)
##############################

# Import Dataset:
data_train <- vroom("./data/train.csv") %>%
  mutate(Cover_Type=factor(Cover_Type))# grab training data

data_test <- vroom("./data/test.csv") # grab testing data

# data_train

################################################################################
# Recipe/Bake

rFormula <- Cover_Type ~ .

fct_recipe <- recipe(Cover_Type ~ ., data = data_train) %>%
  update_role(Id, new_role = "Id") %>%
  # step_mutate_at(c(12:55), fn = factor) %>%
  step_nzv(freq_cut = 15070/50) %>%
  step_zv(all_predictors()) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(Cover_Type)) %>%
  step_normalize(all_numeric_predictors())

prepped_recipe <- prep(fct_recipe) # preprocessing new data
baked_data <- bake(prepped_recipe, new_data = data_train)


################################################################################
########################
##### Stacked mods #####
########################

untuned_model <- control_stack_grid()
tuned_model <- control_stack_resamples()

folds <- vfold_cv(data_train, v = 5, repeats = 1)


# Model 1: Classification RF

class_rf_mod <- rand_forest(mtry = tune(),
                            min_n = tune(),
                            trees = 1000) %>% #Type of model
  set_engine('ranger') %>%
  set_mode('classification')

rf_pretune_wf <- workflow() %>%
  add_recipe(fct_recipe) %>%
  add_model(class_rf_mod)

# Run CV
rf_final_mod <- fit_resamples(rf_pretune_wf,
                              resamples = folds,
                              metrics = metric_set(roc_auc),
                              control = tuned_model)


# Model 2: Light GBM

boost_model <- boost_tree(trees = 300,
                          tree_depth = 4,
                          learn_rate = .05,
                          mtry = 15,
                          min_n = 12
) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

boost_wf <- workflow() %>%
  add_recipe(fct_recipe) %>%
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

boost_final_mod <- fit_resamples(boost_wf,
                          resamples = folds,
                          metrics = metric_set(roc_auc),
                          control = tuned_model)


# Stacked model

models_stack <- 
  stacks() %>%
  add_candidates(rf_final_mod) %>%
  add_candidates(boost_final_mod)


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


# Output as csv

vroom_write(lgbm_predictions_boost, "./data/lgbm_pred_boost.csv", delim = ",")
# save(file = '.RData', list = c('final_wf'))
# load('ggg_nn_wf.RData')




