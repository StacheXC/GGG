library(tidymodels)
library(vroom)
library(embed)

train = vroom("train.csv")
test = vroom("test.csv")
train$type = as.factor(train$type)

my_recipe = recipe(type ~ ., train) |> 
  update_role(id, new_role="id") |> 
  step_mutate(color = as.factor(color)) |> 
  step_dummy(color) |> 
  step_range(all_numeric_predictors(), min=0, max=1)

prepped_recipe = prep(my_recipe)
baked_recipe = bake(prepped_recipe, new_data = train)

my_model = mlp(hidden_units = tune(),
               epochs = 50) |> 
  set_engine("keras") |> 
  set_mode("classification")

wf = workflow() |> 
  add_recipe(my_recipe) |> 
  add_model(my_model)

tuning_grid = grid_regular(hidden_units(range=c(1, 20)),
                           levels=10)

folds = vfold_cv(train, v = 10, repeats = 1)

CV_results = wf |> 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

CV_results |> 
  collect_metrics() |> 
  filter(.metric=="accuracy") |> 
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()


bestTune = CV_results |> 
  select_best()

final_wf = wf |> 
  finalize_workflow(bestTune) |> 
  fit(data = train)

predictions = predict(final_wf,
                      new_data = test,
                      type = "class")

kaggle_submission = bind_cols(test["id"], predictions[".pred_class"]) |> 
  rename("id" = id, "type" = .pred_class)

vroom_write(x = kaggle_submission, 
            file = "nnpred.csv",
            delim = ",")


