library(data.table) # fastest way to read csv files
library(DataExplorer) # create basic EDA
library(tidyverse)
library(tidymodels)
options(tidymodels.dark = TRUE)
library(ggplot2)
theme_set(theme_minimal())
library(ggridges)
library(themis) # dealing with unbalanced data
library(doParallel)
#devtools::install_github("stevenpawley/recipeselectors", force = TRUE)
#library(recipeselectors)
library(finetune)
library(parsnip)
library(treesnip)

# speed up computation with parallel processing
all_cores <- parallel::detectCores(logical = FALSE)
registerDoParallel(cores = all_cores)

cc = fread("creditcard.csv")
str(cc)
cc$Class = as.factor(cc$Class) 

create_report(cc)
# obviously all of the predictors are independent to each other
# no missing values

# calculate the percentage of the respond variable
table(cc$Class)/NROW(cc)*100

# calculate correlation for each predictors with respond variable
cc %>% 
  pivot_longer(!Class, names_to = 'variable', values_to = 'value') %>%
  ggplot(aes(y=Class, x=value, fill=Class)) + geom_density_ridges() +
  facet_wrap(vars(variable), nrow = 6, ncol = 5, scales = "free_x")

# there is a clear difference in density on several variables, especially in v10, v11, v12, v14, v16, v17, and v18
# to prove this assumption, we are going to calculate point bi-serial correlation and plot it
corr_df = data.frame(matrix(data = NA, nrow = NCOL(cc)-1, ncol = 3))
colnames(corr_df) = c('Variable', 'Correlation', 'pvalue')
corr_df$Variable = colnames(cc)[-31]
for (i in 1:(NCOL(cc)-1)) {
  test = cor.test(as.numeric(cc$Class), unlist(cc[,..i]))
  corr_df$Correlation[i] = test$estimate
  corr_df$pvalue[i] = test$p.value
}
# looks like the p-value is very low due to large number of observations
ggplot(corr_df) +
  aes(x = reorder(Variable,Correlation), y = Correlation) +
  geom_col(fill = "#112446") +
  theme_minimal() + labs(x='Variable')

# split the dataset
cc_split <- initial_split(
  select(cc, -Time),
  prop = 0.8,
  strata = Class)

cc_folds = training(cc_split) %>% vfold_cv(v=10, strata = 'Class')
tidy(cc_folds) %>% count(Data)
rm(cc)

# preprocessing and over-sampling
basic_recipe = recipe(Class~., data = training(cc_split)) %>%
  step_nzv(all_predictors()) %>% prep()
smote_recipe = recipe(Class~., data = training(cc_split)) %>%
  step_smote(seed = 123, Class, over_ratio = 0.5) %>%
  step_nzv(all_predictors()) %>% prep()
adasyn_recipe = recipe(Class~., data = training(cc_split)) %>%
  step_adasyn(seed = 135, Class, over_ratio = 0.5) %>%
  step_nzv(all_predictors()) %>% prep()
rose_recipe = recipe(Class~., data = training(cc_split)) %>%
  step_rose(seed = 245, Class, over_ratio = 0.5) %>%
  step_nzv(all_predictors()) %>% prep()
roc_recipe = recipe(Class~., data = training(cc_split)) %>%
  step_select_roc(all_predictors(), outcome = "Class", top_p = NA, threshold = 0.7) %>%
  recipes::prep()
roc_smote_recipe = recipe(Class~., data = training(cc_split)) %>%
  step_smote(seed = 123) %>%
  step_select_roc(all_predictors(), outcome = "Class", top_p = NA, threshold = 0.7) %>%
  step_nzv() %>%
  recipes::prep()
roc_adasyn_recipe = recipe(Class~., data = training(cc_split)) %>%
  step_adasyn(seed = 135) %>%
  step_select_roc(all_predictors(), outcome = "Class", top_p = NA, threshold = 0.7) %>%
  step_nzv() %>%
  recipes::prep()
roc_rose_recipe = recipe(Class~., data = training(cc_split)) %>%
  step_rose(seed = 245) %>%
  step_select_roc(all_predictors(), outcome = "Class", top_p = NA, threshold = 0.7) %>%
  step_nzv() %>%
  recipes::prep()

# specify models
gbm = boost_tree(mtry = tune(), trees = tune(), tree_depth = tune(), 
                 learn_rate = tune(), min_n = tune(), loss_reduction = tune()) %>% 
  set_engine("lightgbm") %>% 
  set_mode("classification")

rf = rand_forest(mtry = tune(),
                 trees = tune(),
                 min_n = tune()) %>% 
  set_engine("randomForest") %>% 
  set_mode("classification")

svm = svm_poly(mode = "classification",
               engine = "kernlab",
               cost = tune(),
               degree = tune(),
               scale_factor = tune())

mlp = mlp(hidden_units = tune(),
          penalty = tune(),
          epochs = tune()) %>%  
  set_engine("nnet") %>% 
  set_mode("classification")

nb = naive_Bayes(mode = "classification",
                 smoothness = tune(),
                 Laplace = tune(),
                 engine = "klaR")

dc = decision_tree(tree_depth = tune(), min_n = tune(), cost_complexity = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

logreg = logistic_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

cc_wf = workflow_set(
  preproc = list('basic'=basic_recipe,
                 'smote' = smote_recipe),
  models = list(svm, mlp, dc, logreg, nb))

ctrl_sa <- control_sim_anneal(verbose = TRUE, no_improve = 10L, save_workflow = FALSE, event_level = 'second',
                              parallel_over = 'everything', save_pred = FALSE)
cc_start = Sys.time()
cc_sa = cc_wf %>% workflow_map('tune_sim_anneal', seed = 31, resamples = cc_folds, 
                               metrics = yardstick::metric_set(pr_auc, f_meas, recall, precision),
                               iter=30, control=ctrl_sa)
cc_end = Sys.time()
cc_end - cc_start
warnings()
show_notes(.Last.tune.result)
cc_sa %>% workflowsets::rank_results() %>% write.csv('Simulated Annealing Result.csv', row.names = F)
cc_sa %>% workflowsets::collect_metrics() %>% filter(.metric=='pr_auc') %>% arrange(-mean)

autoplot(cc_sa, rank_metric = 'pr_auc', metric = 'pr_auc', select_best = T)
autoplot(cc_sa, type='parameters')

# auto-ml
library(agua)
library(pins)
library(vetiver)
library(openxlsx)
h2o::h2o.init()

## h20 set_engine argument : https://rdrr.io/cran/h2o/man/h2o.automl.html
# run for a maximum of 30 minutes
auto_1800 <-
  auto_ml() %>%
  set_engine("h2o", max_runtime_secs = 1800, seed = 1, nfolds=10, sort_metric = 'AUCPR', stopping_metric = "AUCPR") %>%
  set_mode("classification")

auto_3600 <-
  auto_ml() %>%
  set_engine("h2o", max_runtime_secs = 3600, seed = 1, nfolds=10, sort_metric = 'AUCPR', stopping_metric = "AUCPR") %>%
  set_mode("classification")

rec_1800 <-
  recipe(Class ~ ., data = training(cc_split)) %>% 
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors())

rec_3600 <-
  recipe(Class ~ ., data = training(cc_split)) %>% 
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_bsmote(seed = 135, Class, over_ratio = 0.5) %>%
  step_tomek(Class)

wflow_1800 <-
  workflow() %>%
  add_model(auto_1800) %>%
  add_recipe(rec_1800)

wflow_3600 <-
  workflow() %>%
  add_model(auto_3600) %>%
  add_recipe(rec_3600)

fit_1800 <- fit(wflow_1800, data = training(cc_split))
v <- vetiver_model(fit_1800, "fit_1800")
model_board <- board_folder(getwd())
vetiver_pin_write(model_board, v)
model_board
vetiver_pin_read(model_board, "fit_1800")


fit_3600 <- fit(wflow_3600, data = training(cc_split))
w <- vetiver_model(fit_3600, "fit_3600")
modelw_board <- board_folder(getwd())
vetiver_pin_write(modelw_board, w)
modelw_board
vetiver_pin_read(modelw_board, "fit_3600")


extract_fit_parsnip(fit_1800)

# the result 
rank_results(fit_1800) %>% 
  filter(.metric == "pr_auc") %>%
  arrange(rank) # mean pr_auc 0.849 StackedEnsemble_BestOfFamily_3_AutoML_1_20221113_51204
rank_results(fit_1800) %>% write.xlsx('rank_results_fit_1800.xlsx', overwrite = T)

# pr_auc value across 10 folds
collect_metrics(fit_1800, summarize = FALSE) %>% filter(.metric == 'pr_auc' & algorithm == 'stacking')
collect_metrics(fit_1800, summarize = FALSE) %>% write.xlsx('collect_metrics_fit_1800.xlsx', overwrite = T)

fit_1800[["fit"]][["fit"]][["fit"]]@leader

autoplot(fit_1800, type = "rank", metric = 'pr_auc') +
  theme(legend.position = "none")
#############################################################################################################################
#               1st approach, using pr_auc metric as kaggle suggested
id_1800 = fit_1800[["fit"]][["fit"]][["fit"]]@leader@model_id
pred_1800 = cbind(testing(cc_split) %>% select(Class), 
                 predict(fit_1800, id=id_1800, 
                         new_data = testing(cc_split), type = 'prob'),
                 predict(fit_1800, id=id_1800, 
                         new_data = testing(cc_split)))
caret::confusionMatrix(pred_1800$.pred_class, pred_1800$Class, mode='prec_recall', positive = '1')
## precision 0.932 : 5 transactions are considered fraud even though they are not fraud, no money is lost but customers are likely to be angry
## recall 0.767 : 21 transactions is predicted as not fraud even tough it is a real fraud, hence the customer lose the money
recall(pred_1800, truth = Class, .pred_class, event_level = 'second')
precision(pred_1800, truth = Class, .pred_class, event_level = 'second')
pr_auc(pred_1800, truth = Class, .pred_p1, event_level = 'second')
autoplot(pr_curve(pred_1800, truth = Class, .pred_p1,event_level = 'second')) +
  labs(title = id_1800,
       subtitle = paste('pr_auc testing:', 
                        round(pr_auc(pred_1800, truth = Class, .pred_p1, event_level = 'second')[3],3)))

#############################################################################################################################
#               2nd approach, using recall@0.9 to maximize precision while maintain recall at 0.9
train_pred_1800 = cbind(training(cc_split) %>% select(Class), 
                  predict(fit_1800, id=id_1800, 
                          new_data = training(cc_split), type = 'prob'),
                  predict(fit_1800, id=id_1800, 
                          new_data = training(cc_split)))
pr_auc(train_pred_1800, truth = Class, .pred_p1, event_level = 'second') # it's so excellent because this metric is measured using all training data
rec_prec_threshold_1800 = data.frame(Threshold = seq(0,1,length.out = 1000),
                                     Recall = NA,
                                     Precision = NA)
for (i in 1:NROW(rec_prec_threshold_1800)) {
  pred = factor(ifelse(train_pred_1800$.pred_p1>rep(rec_prec_threshold_1800[i,1], NROW(train_pred_1800)), 1, 0), levels = c(0,1))
  rec_prec_threshold_1800$Recall[i] = recall_vec(truth = train_pred_1800$Class, estimate = pred, event_level = 'second')
  rec_prec_threshold_1800$Precision[i] = precision_vec(truth = train_pred_1800$Class, estimate = pred, event_level = 'second')
}

for (i in 1:NROW(rec_prec_threshold_1800)) {
  pred = factor(ifelse(test_pred_1800$.pred_p1>rep(rec_prec_threshold_1800[i,1], NROW(test_pred_1800)), 1, 0), levels = c(0,1))
  rec_prec_threshold_1800$Recall[i] = recall_vec(truth = test_pred_1800$Class, estimate = pred, event_level = 'second')
  rec_prec_threshold_1800$Precision[i] = precision_vec(truth = test_pred_1800$Class, estimate = pred, event_level = 'second')
}

head(rec_prec_threshold_1800)
data = rec_prec_threshold_1800

precision_recall_plot = function(data, recall.point=0.9, precision.point=0.9) {
  print(
    data %>% pivot_longer(Recall:Precision, names_to = 'Metric') %>% unnest(value) %>%
      ggplot(aes(x=Threshold, y=value, col=Metric)) + geom_line() + 
      # recall at threshold point
      geom_vline(xintercept = data[which.min(abs(unlist(data$Recall)-recall.point)),1], col='grey') +
      annotate('text', x=(data[which.min(abs(unlist(data$Recall)-recall.point)),1]-data[which.min(abs(unlist(data$Recall)-recall.point))+15,1]), 
               y=0.1, label=paste('Recall@ ', round(unlist(data[which.min(abs(unlist(data$Recall)-recall.point)),2]),3)), 
               angle=90) +
      annotate('text', x=data[which.min(abs(unlist(data$Recall)-recall.point))+15,1], 
               y=0.1, label=paste('Precision: ', round(unlist(data[which.min(abs(unlist(data$Recall)-recall.point)),3]),3)), 
               angle=90) +
      # precision at threshold point
      geom_vline(xintercept = data[which.min(abs(unlist(data$Precision)-precision.point)),1], col='grey') +
      annotate('text', x=data[which.min(abs(unlist(data$Precision)-precision.point))-15,1], 
               y=0.9, label=paste('Precision@ ', round(unlist(data[which.min(abs(unlist(data$Precision)-precision.point)),3]),3)), 
               angle=90) +
      annotate('text', x=data[which.min(abs(unlist(data$Precision)-precision.point))+15,1], 
               y=0.9, label=paste('Recall: ', round(unlist(data[which.min(abs(unlist(data$Precision)-precision.point)),2]),3)), 
               angle=90) +
      # threshold point where recall = precision
      geom_vline(xintercept = data[which.min(abs(unlist(data$Precision)-unlist(data$Recall))),1], col='grey', linetype='dashed') +
      annotate('text', x=data[which.min(abs(unlist(data$Precision)-unlist(data$Recall)))-15,1], 
               y=0.5, label=paste('Precision: ', round(unlist(data[which.min(abs(unlist(data$Precision)-unlist(data$Recall))),3]),3)), 
               angle=90) +
      annotate('text', x=data[which.min(abs(unlist(data$Precision)-unlist(data$Recall)))+15,1], 
               y=0.5, label=paste('Recall: ', round(unlist(data[which.min(abs(unlist(data$Precision)-unlist(data$Recall))),3]),3)), 
               angle=90)
  )
  out = list(recall.thres = data[which.min(abs(unlist(data$Recall)-recall.point)),1],
             precision.thres = data[which.min(abs(unlist(data$Precision)-precision.point)),1],
             balance.thres = data[which.min(abs(unlist(data$Precision)-unlist(data$Recall))),1])
  return(out)
}
precision_recall_plot(rec_prec_threshold_1800, 0.90, 0.90)
train_pred_1800$.pred_class_recall_0.9 = factor(ifelse(train_pred_1800$.pred_p1>rep(0.1181181, NROW(train_pred_1800)), 1, 0), levels = c(0,1))
caret::confusionMatrix(train_pred_1800$.pred_class_recall_0.9, train_pred_1800$Class, mode='prec_recall', positive = '1')

test_pred_1800 = cbind(testing(cc_split) %>% select(Class), 
                        predict(fit_1800, id=id_1800, 
                                new_data = testing(cc_split), type = 'prob'))
head(test_pred_1800)
test_pred_1800$.pred_class_recall_0.9 = factor(ifelse(test_pred_1800$.pred_p1>rep(0.002002002, NROW(test_pred_1800)), 1, 0), levels = c(0,1))
caret::confusionMatrix(test_pred_1800$.pred_class_recall_0.9, test_pred_1800$Class, mode='prec_recall', positive = '1')
## recall 0.844 : 14 transactions is predicted as not fraud even tough it is a real fraud, 
## obviously an improvement since our 1st approach used pr_auc metric
## we need to implement this with CV to obtain the best threshold

#############################################################################################################################
#               3rd approach, minimize money lost due to fraud transactions by moving the threshold
# let's assume that:
# prediction == true, reference == false, no fraud, but the CS will have to resolve the situation with the customer, CS's effort is worth 10% of the transaction
# prediction == false, reference == true, fraud transaction --> customer lost the money
money_lost = data.frame(Threshold = seq(0,1,length.out = 1001),
                        Lost = rep(NA, 1001))
#money = rep(NA, NROW(train_pred_1800))
for (i in 1:NROW(money_lost)) {
  pred = factor(ifelse(test_pred_1800$.pred_p1>rep(money_lost[i,1], NROW(test_pred_1800)), 1, 0), levels = c(0,1))
  df = testing(cc_split) %>% select(Amount, Class) %>% bind_cols('Prediction'=pred, 'Potential_Lost'=0) %>%
    filter((Class==0 & Prediction==1) | (Class==1 & Prediction==0))
  df$Potential_Lost=ifelse(df$Class==0,df$Amount/10,df$Amount)
  if (numbers::mod(i,50)==0) {
    print(i)
  }
  money_lost$Lost[i] = -sum(df$Potential_Lost)
}
money_lost[which.max(money_lost$Lost),1]
ggplot(money_lost) +
  aes(x = Threshold, y = Lost) +
  geom_line(colour='black') +
  ylim(-30000,0) +
  geom_vline(xintercept = money_lost[which.max(money_lost$Lost),1], linetype="dashed", col="grey20") +
  labs(title = 'Potential Fraud Transaction Amount', 
       subtitle = paste('Minimum money lost is', money_lost[which.max(money_lost$Lost),2], 'at','Threshold',money_lost[which.max(money_lost$Lost),1])) +
  theme_minimal()

# calculate amount of many lost in testing
testing(cc_split) %>% select(Amount, Class) %>% 
  bind_cols(Prediction=factor(ifelse(test_pred_1800$.pred_p1>rep(money_lost[which.max(money_lost$Lost),1], NROW(test_pred_1800)), 1, 0), levels = c(0,1))) %>%
  filter((Class==0 & Prediction==1) | (Class==1 & Prediction==0)) %>% 
  mutate(Amount_Lost = ifelse(Class==0,Amount/10,Amount)) %>%
  select(Amount_Lost) %>% sum()

# calculate recall at threshold 0.108 for training data
train_pred_1800_at_money = train_pred_1800 %>% select(Class:.pred_p1) %>% 
  bind_cols(.pred_at_money=factor(ifelse(train_pred_1800$.pred_p1>rep(money_lost[which.max(money_lost$Lost),1], NROW(train_pred_1800)), 1, 0), levels = c(0,1)))
caret::confusionMatrix(train_pred_1800_at_money$.pred_at_money, train_pred_1800_at_money$Class, 
                       mode='prec_recall', positive = '1')

# calculate recall at threshold 0.108 for testing data
test_pred_1800_at_money = test_pred_1800 %>% select(Class:.pred_p1) %>% 
  bind_cols(.pred_at_money=factor(ifelse(test_pred_1800$.pred_p1>rep(money_lost[which.max(money_lost$Lost),1], NROW(test_pred_1800)), 1, 0), levels = c(0,1)))
caret::confusionMatrix(test_pred_1800_at_money$.pred_at_money, test_pred_1800_at_money$Class, 
                       mode='prec_recall', positive = '1') 

# calculate amount of money lost with 1st and 2nd approach
testing(cc_split) %>% select(Amount, Class) %>% 
  mutate('Prediction'=predict(fit_1800, id=id_1800,new_data = testing(cc_split))) %>%
  filter((Class==0 & Prediction==1) | (Class==1 & Prediction==0)) %>% 
  mutate(Amount_Lost = ifelse(Class==0,Amount/10,Amount)) %>%
  select(Amount_Lost) %>% sum()

testing(cc_split) %>% select(Amount, Class) %>% 
  bind_cols(Prediction=factor(ifelse(test_pred_1800$.pred_p1>rep(0.1181181, NROW(test_pred_1800)), 1, 0), levels = c(0,1))) %>%
  filter((Class==0 & Prediction==1) | (Class==1 & Prediction==0)) %>% 
  mutate(Amount_Lost = ifelse(Class==0,Amount/10,Amount)) %>%
  select(Amount_Lost) %>% sum()

#############################################################################################################################
extract_fit_parsnip(fit_3600)

# the result 
rank_results(fit_3600) %>% 
  filter(.metric == "pr_auc") %>%
  arrange(rank) # 10 1st model achieved pr_auc almost 1 (it shows 1 due to round function). This results are too good to be true
# i'll just use the 1st one
fit_3600[["fit"]][["fit"]][["fit"]]@leader

# pr_auc value across 10 folds
collect_metrics(fit_3600, summarize = FALSE) %>% filter(.metric == 'pr_auc' & algorithm == 'stacking') %>% View()

rank_results(fit_3600) %>% write.xlsx('rank_results_fit_3600.xlsx', overwrite = T)
collect_metrics(fit_3600, summarize = FALSE) %>% write.xlsx('collect_metrics_fit_3600.xlsx', overwrite = T)

autoplot(fit_3600, type = "rank", metric = 'pr_auc') +
  theme(legend.position = "none")
# i think random forest is the most stable algorithm than gradient boosting even though 2nd and 3rd best models are gb

#############################################################################################################################
#               1st approach, using pr_auc metric as kaggle suggested
id_3600 = fit_3600[["fit"]][["fit"]][["fit"]]@leader@model_id
pred_3600 = cbind(testing(cc_split) %>% select(Class), 
                  predict(fit_3600, id=id_3600, 
                          new_data = testing(cc_split), type = 'prob'),
                  predict(fit_3600, id=id_3600, 
                          new_data = testing(cc_split)))
caret::confusionMatrix(pred_3600$.pred_class, pred_3600$Class, mode='prec_recall', positive = '1')
## precision 0.945946 : 4 transactions are considered fraud even though they are not fraud, no money is lost but customers are likely to be angry
## recall 0.777778 : 20 transactions is predicted as not fraud even tough it is a real fraud, hence the customer lose the money
## this predictions is slightly worse than before, i guess adasyn cant help much 
recall(pred_3600, truth = Class, .pred_class, event_level = 'second')
precision(pred_3600, truth = Class, .pred_class, event_level = 'second')
pr_auc(pred_3600, truth = Class, .pred_p1, event_level = 'second') # same auc_pr
autoplot(pr_curve(pred_3600, truth = Class, .pred_p1,event_level = 'second')) +
  labs(title = id_3600,
       subtitle = paste('pr_auc testing:', 
                        round(pr_auc(pred_3600, truth = Class, .pred_p1, event_level = 'second')[3],3)))

#############################################################################################################################
#               2nd approach, using recall@0.9 to maximize precision while maintain recall at 0.9
train_pred_3600 = cbind(training(cc_split) %>% select(Class), 
                        predict(fit_3600, id=id_3600, 
                                new_data = training(cc_split), type = 'prob'),
                        predict(fit_3600, id=id_3600, 
                                new_data = training(cc_split)))
pr_auc(train_pred_3600, truth = Class, .pred_p1, event_level = 'second') # it's so excellent because this metric is measured using all training data
# this value is to be expected since the mean pr_auc is almost 1
rec_prec_threshold_3600 = data.frame(Threshold = seq(0,1,length.out = 1000),
                                     Recall = NA,
                                     Precision = NA)
for (i in 1:NROW(rec_prec_threshold_3600)) {
  pred = factor(ifelse(train_pred_3600$.pred_p1>rep(rec_prec_threshold_3600[i,1], NROW(train_pred_3600)), 1, 0), levels = c(0,1))
  rec_prec_threshold_3600$Recall[i] = recall_vec(truth = train_pred_3600$Class, estimate = pred, event_level = 'second')
  rec_prec_threshold_3600$Precision[i] = precision_vec(truth = train_pred_3600$Class, estimate = pred, event_level = 'second')
}

for (i in 1:NROW(rec_prec_threshold_3600)) {
  pred = factor(ifelse(test_pred_3600$.pred_p1>rep(rec_prec_threshold_3600[i,1], NROW(test_pred_3600)), 1, 0), levels = c(0,1))
  rec_prec_threshold_3600$Recall[i] = recall_vec(truth = test_pred_3600$Class, estimate = pred, event_level = 'second')
  rec_prec_threshold_3600$Precision[i] = precision_vec(truth = test_pred_3600$Class, estimate = pred, event_level = 'second')
}

head(rec_prec_threshold_3600)
data = rec_prec_threshold_1800

precision_recall_plot = function(data, recall.point=0.9, precision.point=0.9) {
  print(
    data %>% pivot_longer(Recall:Precision, names_to = 'Metric') %>% unnest(value) %>%
      ggplot(aes(x=Threshold, y=value, col=Metric)) + geom_line() + 
      # recall at threshold point
      geom_vline(xintercept = data[which.min(abs(unlist(data$Recall)-recall.point)),1], col='grey') +
      annotate('text', x=data[which.min(abs(unlist(data$Recall)-recall.point))-15,1], 
               y=0.1, label=paste('Recall@ ', round(unlist(data[which.min(abs(unlist(data$Recall)-recall.point)),2]),3)), 
               angle=90) +
      annotate('text', x=data[which.min(abs(unlist(data$Recall)-recall.point))+15,1], 
               y=0.1, label=paste('Precision: ', round(unlist(data[which.min(abs(unlist(data$Recall)-recall.point)),3]),3)), 
               angle=90) +
      # precision at threshold point
      geom_vline(xintercept = data[which.min(abs(unlist(data$Precision)-precision.point)),1], col='grey') +
      annotate('text', x=data[which.min(abs(unlist(data$Precision)-precision.point))-15,1], 
               y=0.9, label=paste('Precision@ ', round(unlist(data[which.min(abs(unlist(data$Precision)-precision.point)),3]),3)), 
               angle=90) +
      annotate('text', x=data[which.min(abs(unlist(data$Precision)-precision.point))+15,1], 
               y=0.9, label=paste('Recall: ', round(unlist(data[which.min(abs(unlist(data$Precision)-precision.point)),2]),3)), 
               angle=90) +
      # threshold point where recall = precision
      geom_vline(xintercept = data[which.min(abs(unlist(data$Precision)-unlist(data$Recall))),1], col='grey', linetype='dashed') +
      annotate('text', x=data[which.min(abs(unlist(data$Precision)-unlist(data$Recall)))-15,1], 
               y=0.5, label=paste('Precision: ', round(unlist(data[which.min(abs(unlist(data$Precision)-unlist(data$Recall))),3]),3)), 
               angle=90) +
      annotate('text', x=data[which.min(abs(unlist(data$Precision)-unlist(data$Recall)))+15,1], 
               y=0.5, label=paste('Recall: ', round(unlist(data[which.min(abs(unlist(data$Precision)-unlist(data$Recall))),3]),3)), 
               angle=90)
  )
  out = list(recall.thres = data[which.min(abs(unlist(data$Recall)-recall.point)),1],
             precision.thres = data[which.min(abs(unlist(data$Precision)-precision.point)),1],
             balance.thres = data[which.min(abs(unlist(data$Precision)-unlist(data$Recall))),1])
  return(out)
}
precision_recall_plot(rec_prec_threshold_3600, 0.90, 0.90) # good precision but low recall (?)
train_pred_3600$.pred_class_recall_0.9 = factor(ifelse(train_pred_3600$.pred_p1>rep(0.3233233, NROW(train_pred_3600)), 1, 0), levels = c(0,1))
caret::confusionMatrix(train_pred_3600$.pred_class_recall_0.9, train_pred_3600$Class, mode='prec_recall', positive = '1')

test_pred_3600 = cbind(testing(cc_split) %>% select(Class), 
                       predict(fit_3600, id=id_3600, 
                               new_data = testing(cc_split), type = 'prob'))
head(test_pred_3600)
test_pred_3600$.pred_class_recall_0.9 = factor(ifelse(test_pred_3600$.pred_p1>rep(0.004004004, NROW(test_pred_3600)), 1, 0), levels = c(0,1))
caret::confusionMatrix(test_pred_3600$.pred_class_recall_0.9, test_pred_3600$Class, mode='prec_recall', positive = '1')
## recall 0.82 : it's okay, but not good enough, 16 out of 74 customers lost their money due to fraud  

#############################################################################################################################
#               3rd approach, minimize money lost due to fraud transactions by moving the threshold
# let's assume that:
# prediction == true, reference == false, no fraud, but the CS will have to resolve the situation with the customer, CS's effort is worth 10% of the transaction
# prediction == false, reference == true, fraud transaction --> customer lost the money
money_lost_3600 = data.frame(Threshold = seq(0,1,length.out = 1001),
                        Lost = rep(NA, 1001))
#money = rep(NA, NROW(train_pred_1800))
for (i in 1:NROW(money_lost_3600)) {
  pred = factor(ifelse(test_pred_3600$.pred_p1>rep(money_lost[i,1], NROW(test_pred_3600)), 1, 0), levels = c(0,1))
  df = testing(cc_split) %>% select(Amount, Class) %>% bind_cols('Prediction'=pred, 'Potential_Lost'=0) %>%
    filter((Class==0 & Prediction==1) | (Class==1 & Prediction==0))
  df$Potential_Lost=ifelse(df$Class==0,df$Amount/10,df$Amount)
  if (numbers::mod(i,50)==0) {
    print(i)
  }
  money_lost_3600$Lost[i] = -sum(df$Potential_Lost)
}

ggplot(money_lost_3600) +
  aes(x = Threshold, y = Lost) +
  geom_line(colour='black') +
  ylim(-50000,0) +
  geom_vline(xintercept = money_lost[which.max(money_lost_3600$Lost),1], linetype="dashed", col="grey20") +
  labs(title = 'Potential Fraud Transaction Amount', 
       subtitle = paste('Minimum money lost is', 
                        money_lost_3600[which.max(money_lost_3600$Lost),2], 'at',
                        'Threshold',money_lost_3600[which.max(money_lost_3600$Lost),1])) +
  theme_minimal() # worse than the first attempt (without over-sampling)

# calculate amount of many lost in testing
testing(cc_split) %>% select(Amount, Class) %>% 
  bind_cols(Prediction=factor(ifelse(test_pred_3600$.pred_p1>rep(money_lost_3600[which.max(money_lost_3600$Lost),1], NROW(test_pred_3600)), 1, 0), levels = c(0,1))) %>%
  filter((Class==0 & Prediction==1) | (Class==1 & Prediction==0)) %>% 
  mutate(Amount_Lost = ifelse(Class==0,Amount/10,Amount)) %>%
  select(Amount_Lost) %>% sum()

# calculate recall at threshold 0.108 for training data
train_pred_3600_at_money = train_pred_3600 %>% select(Class:.pred_p1) %>% 
  bind_cols(.pred_at_money=factor(ifelse(train_pred_3600$.pred_p1>rep(money_lost_3600[which.max(money_lost_3600$Lost),1], NROW(train_pred_3600)), 1, 0), levels = c(0,1)))
caret::confusionMatrix(train_pred_3600_at_money$.pred_at_money, train_pred_3600_at_money$Class, 
                       mode='prec_recall', positive = '1') # got recall 0.947761
# calculate recall at threshold 0.108 for testing data
test_pred_3600_at_money = test_pred_3600 %>% select(Class:.pred_p1) %>% 
  bind_cols(.pred_at_money=factor(ifelse(test_pred_3600$.pred_p1>rep(money_lost_3600[which.max(money_lost_3600$Lost),1], NROW(test_pred_3600)), 1, 0), levels = c(0,1)))
caret::confusionMatrix(test_pred_3600_at_money$.pred_at_money, test_pred_3600_at_money$Class, 
                       mode='prec_recall', positive = '1') # got recall 0.833333 

# calculate amount of money lost with 1st and 2nd approach
testing(cc_split) %>% select(Amount, Class) %>% 
  mutate('Prediction'=predict(fit_3600, id=id_3600,new_data = testing(cc_split))) %>%
  filter((Class==0 & Prediction==1) | (Class==1 & Prediction==0)) %>% 
  mutate(Amount_Lost = ifelse(Class==0,Amount/10,Amount)) %>%
  select(Amount_Lost) %>% sum()

testing(cc_split) %>% select(Amount, Class) %>% 
  bind_cols(Prediction=factor(ifelse(test_pred_3600$.pred_p1>rep(0.004004004, NROW(test_pred_3600)), 1, 0), levels = c(0,1))) %>%
  filter((Class==0 & Prediction==1) | (Class==1 & Prediction==0)) %>% 
  mutate(Amount_Lost = ifelse(Class==0,Amount/10,Amount)) %>%
  select(Amount_Lost) %>% sum()

h2o::h2o.shutdown()

