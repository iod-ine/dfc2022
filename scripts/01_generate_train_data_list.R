# Generate a CSV with every file in the training set.

library(tidyverse)

data_dir <- "~/Projects/Data/DFC2022"

labeled_train <- file.path(data_dir, "labeled_train")
unlabeled_train <- file.path(data_dir, "unlabeled_train")

labeled_files <- list.files(
  labeled_train,
  recursive = TRUE,
  pattern = "E\\d{3}\\.tif$"
)

unlabeled_files <- list.files(
  unlabeled_train,
  recursive = TRUE,
  pattern = "E\\d{3}\\.tif$"
)

labeled_index <- tibble(
  path = labeled_files,
  labeled = "yes"
)

unlabeled_index <- tibble(
  path = unlabeled_files,
  labeled = "no"
)

combined_index <- rbind(labeled_index, unlabeled_index)

combined_index <- combined_index |>
  separate(path, c("region", "type", "filename"), sep = "/") |>
  select(-type)

write_csv(combined_index, "misc/training_data.csv")
