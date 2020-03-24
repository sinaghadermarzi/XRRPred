library(keras)

train <- read.csv("training_features_imputed.txt", sep = "\t")
test <- read.csv("test_features_imputed.txt", sep="\t")

excluded = c("OUT_rfree")
target_col = c("OUT_resolution")
feature_cols = !names(train) %in% c(excluded,target_col)
x_train = data.matrix(train[feature_cols])
y_train = data.matrix(train[target_col])
x_test = data.matrix(test[feature_cols])
y_test = data.matrix(test[target_col])


x_train = scale(x_train)

col_means_train <- attr(x_train, "scaled:center")
col_stddevs_train <- attr(x_train, "scaled:scale")
x_test = scale(x_test, center = col_means_train, scale = col_stddevs_train)



build_model <- function() {
  
  model <- keras_model_sequential() %>%
    layer_dense(units = 512, activation = "relu",
                input_shape = dim(data)[2]) %>%
    layer_dense(units = 256, activation = "relu") %>%
    layer_dense(units = 128, activation = "relu") %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 8, activation = "relu") %>%
    layer_dense(units = 1)
  
  model %>% compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(),
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

epochs <- 5
# Fit the model and store training stats
history <- model %>% fit(
  x_train,
  y_train,
  epochs = epochs,
  validation_split = 0.2,
)
#keras.callbacks.
plot(history)
library(stringi)
code = stringi::stri_rand_strings(1,4,"[0-9]")
model_to_saved_model(model,paste("model_dump_",code))
test_predictions <- model %>% predict(x_test)
test_predictions[ , 1]
corr = cor(test_predictions,y_test,method = "spearman")