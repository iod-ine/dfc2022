# Explore the statistics of the train data.
# The statistics are calculated in the [Data] Construct data distributions
# Jupyter notebook

source("scripts/colors.R")

library(tidyverse)

elevation <- read_csv("misc/stat_elevation_histogram.csv")
rgb <- read_csv("misc/stat_rgb_values_counts.csv")
labels <- read_csv("misc/stat_label_class_count.csv")

labels <- labels |> add_row(class = 8, count = 0)
labels <- labels |> add_row(class = 9, count = 0)

elevation |>
  filter(between(upper, -50, 3000)) |>
  summarise(mean = mean())
  ggplot() +
  geom_bar(mapping = aes(upper, count), stat = "identity")

labels |>
  ggplot() +
  geom_bar(aes(class, count, fill = (factor(class))), stat = "identity") +
  scale_fill_manual(values = urban_atlas_dfc22_colormap)

rgb |>
  ggplot() +
  geom_line(aes(value, count, color = factor(channel)))
