library(tidymodels)
library(GA)
library(doParallel)

data = data.table::fread('D:/My Scripts/Github/Credit Crad Fraud Detection/creditcard.csv')
data = data %>%
  filter(Class==0) %>%
  sample_n(50000) %>%
  bind_rows(data %>%
              filter(Class==1)) %>%
  select(-Time)
str(data)
data$Class = as.factor(data$Class)
data_split = initial_split(data, strata = Class)

eval_func_rf = function(x1, x2, x3, split_data, threshold=0.9) {
  my_model = rand_forest(mtry = round(x1),
                   trees = round(x2),
                   min_n = round(x3)) %>% 
    set_engine("randomForest") %>% 
    set_mode("classification")
  
  my_wflow <-
    workflow() %>%
    add_model(my_model) %>%
    add_formula(Class~.)
  
  my_fit = fit(my_wflow, training(split_data))
  my_pred = predict(my_fit, testing(split_data), type = 'prob')
  
  metric = pr_curve(data.frame(my_pred, Class = testing(split_data)$Class), 
                                 truth = Class, .pred_1, event_level = 'second') %>% 
    mutate(recall_dif = abs(recall-threshold)) %>%
    arrange(recall_dif) %>%
    slice(n=1) %>%
    select(.threshold, precision) %>% 
    unlist() %>%
    as.numeric()
  #print(paste('Threshold: ', metric[1]))
  return(metric[2]) # maximize precision
}
min_max_mtry = c(1, NCOL(data)-1)
min_max_trees = c(10, 2000)
min_max_min_n = c(5, 1e+4)

all_cores <- parallel::detectCores(logical = FALSE) # it uses 32 gb of ram, so I might as well give them all cores

GA_model = ga(type = 'real-valued',
              fitness = function(x) eval_func_rf(x[1], x[2], x[3], data_split, 0.9),
              lower = c(min_max_mtry[1], min_max_trees[1], min_max_min_n[1]),
              upper = c(min_max_mtry[2], min_max_trees[2], min_max_min_n[2]),
              popSize = 50,
              maxiter = 20,
              run = 5,
              suggestions = c(5, 500, 10),
              optim = T,
              keepBest = T,
              seed = 10,
              parallel = all_cores
              )
summary(GA_model)
plot.ga(GA_model, main = "Genetic Algorithm: Precision values at each iteration")
