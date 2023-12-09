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

# fct_recipe <- recipe(Cover_Type ~ ., data = data_train) %>%
#   update_role(Id, new_role = "Id") %>%
#   # step_mutate_at(c(12:55), fn = factor) %>%
#   step_nzv(freq_cut = 15070/50) %>%
#   step_zv(all_predictors()) %>%
#   step_lencode_glm(all_nominal_predictors(), outcome = vars(Cover_Type)) #%>%
#   #step_normalize(all_numeric_predictors())

# fct_recipe <- recipe(Cover_Type ~ ., data = data_train) %>%
#   update_role(Id, new_role = "Id") %>%
#   step_mutate(Id = factor(Id)) %>%
#   step_mutate_at(all_outcomes(), fn = factor, skip = TRUE) %>%
#   step_zv(all_predictors()) #%>%
#   #step_lencode_glm(all_nominal_predictors(), outcome = vars(Cover_Type))

fct_recipe <- recipe(rFormula, data = data_train) %>%
  step_zv(all_predictors()) %>%
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
                            trees = 500) %>% #Type of model
  set_engine('ranger') %>%
  set_mode('classification')

rf_pretune_wf <- workflow() %>%
  add_recipe(fct_recipe) %>%
  add_model(class_rf_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(mtry(range = c(2,ncol(data_train)-1)),
                            min_n(),
                            levels = 3) ## L^2 total tuning possibilities

# Run CV
rf_final_mod <- rf_pretune_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune <- rf_final_mod %>%
  select_best('roc_auc')

rf_results1 <- rf_pretune_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = data_train)


# Model 2: xg boost

xgboost_recipe <- recipe(rFormula, data = data_train) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

boost_model <- boost_tree(trees = 500,
                          tree_depth = 8,
                          learn_rate = .2
) %>%
  set_engine("xgboost") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

boost_wf <- workflow() %>%
  add_recipe(xgboost_recipe) %>%
  add_model(boost_model)

# Run CV
# tuned_boost <- boost_wf %>%
#   tune_grid(resamples = folds,
#             grid = boost_tuneGrid,
#             metrics = metric_set(accuracy))
# 
# bestTune <- tuned_boost %>%
#   select_best('accuracy')

boost_results1 <- fit_resamples(boost_wf,
                          resamples = folds,
                          metrics = metric_set(roc_auc),
                          control = tuned_model)


# Model 3: Neural Nets

nn_recipe <- recipe(Cover_Type~., data = data_train) %>%
  step_rm(Id) %>%
  step_zv(all_numeric_predictors()) %>%
  step_range(all_numeric_predictors(), min=0, max=1)

nn_model <- mlp(hidden_units = tune(),
                epochs = 50) %>%
  set_engine('nnet') %>%
  set_mode('classification')

nn_wf <- workflow %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

nn_tuneGrid <- grid_regular(hidden_units(range=c(1,10)),
                            levels=5)

# Run CV
tuned_nn <- nn_wf %>%
  tune_grid(resamples = folds,
            grid = nn_tuneGrid,
            metrics = metric_set(roc_auc))

bestTune_nn <- tuned_nn %>%
  select_best('roc_auc')

nn_results1 <- nn_wf %>%
  finalize_workflow(bestTune_nn) %>%
  fit(data = data_train)


# Output as csv

# vroom_write(lgbm_predictions_boost, "./data/lgbm_pred_boost.csv", delim = ",")
save(file = 'RFFinal.RData', list = c('rf_results1'))
save(file = 'BoostFinal.RData', list = c('boost_results1'))
save(file = 'NNFinal.RData', list = c('nn_results1 '))



