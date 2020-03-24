library(keras)


mae <- function(true,pred)
{
  error <- true - pred
  mean(abs(error))
}


train <- read.csv("train_set_resolution.txt",sep = "\t")
test <- read.csv("test_set_resolution.txt", sep= "\t")

excluded = c()
target_col = c("OUTPUT_Resolution")
feature_cols = !names(train) %in% c(excluded,target_col)
x_train = data.matrix(train[feature_cols])
y_train = data.matrix(train[target_col])
x_test = data.matrix(test[feature_cols])
y_test = data.matrix(test[target_col])


# x_train = scale(x_train)
# 
# col_means_train <- attr(x_train, "scaled:center")
# col_stddevs_train <- attr(x_train, "scaled:scale")
# x_test = scale(x_test, center = col_means_train, scale = col_stddevs_train)



build_model <- function() {
  
  model <- keras_model_sequential() %>%
    layer_dense(units = 256, activation = "relu",
                input_shape = dim(data)[2]) %>%
    layer_dropout(0.6) %>%
    # layer_dense(units = 128, activation = "relu") %>%
    # layer_dropout(0.3) %>%
    # layer_dense(units = 64, activation = "relu") %>%
    # layer_dropout(0.3) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dropout(0.6) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dropout(0.6) %>%
    layer_dense(units = 1)
  
  model %>% compile(
    loss = "mse",
    # optimizer = optimizer_rmsprop(),
    optimizer = optimizer_adam(),
    metrics = list("mean_absolute_error")
  )
  
  model
}

model <- build_model()


print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)    

epochs <- 200
# Fit the model and store training stats
history <- model %>% fit(
  x_train,
  y_train,
  epochs = epochs,
  validation_split = 0.2,
  # verbose = 0,
  # callbacks = list(print_dot_callback)
)

plot(history)
# model_to_saved_model(model,"model_dump")
test_predictions <- model %>% predict(x_test)
test_predictions[ , 1]
corr = cor(test_predictions,y_test,method = "spearman")
err = mae(test_predictions,y_test)

