source("scripts/colors.R")
source("scripts/plotting_functions.R")

library(tmap)
library(terra)
library(tidyverse)

predictions_path <- "predictions0/"
data_path <- "/home/dubrovin/Projects/Data/DFC2022"

predictions <- list.files(
  predictions_path,
  recursive = TRUE,
  pattern = "_prediction.tif$"
)

df <- tibble(path=predictions) |>
  separate(path, into = c("region", "prediction_file"), sep = "/") |>
  mutate(file = str_replace(
    prediction_file,
    pattern = "_prediction\\.tif",
    replacement = ".tif"
  ))

plot_random_prediction <- function() {
  index <- floor(runif(1, min = 1, max = nrow(df)))

  region <- df[["region"]][[index]]
  file <- df[["file"]][[index]]
  pred_file <- df[["prediction_file"]][[index]]

  pred_file <- file.path(predictions_path, region, pred_file)

  maps <- list(
    plot_tci(file.path(
      data_path,
      "val",
      region,
      "BDORTHO",
      file
    )),
    plot_dem(file.path(
      data_path,
      "val",
      region,
      "RGEALTI",
      str_replace(file, "\\.tif", "_RGEALTI.tif")
    )),
    plot_mask(pred_file),
    plot_aux(pred_file, title = str_remove(file, "\\.tif"))
  )

  tmap_arrange(maps, ncol = 2)
}

plot_random_prediction()
