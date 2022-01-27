library(tibble)

urban_atlas_dfc22_palette <- tibble(
  class = c(
    "No information",
    "Urban fabric",
    "Industrial, commercial, public, military, private and transport units",
    "Mine, dump and construction sites",
    "Artificial non-agricultural vegetated areas",
    "Arable land (annual crops)",
    "Permanent crops",
    "Pastures",
    "Complex and mixed cultivation patterns",
    "Orchards at the fringe of urban classes",
    "Forests",
    "Herbaceous vegetation associations",
    "Open spaces with little or no vegetation",
    "Wetlands",
    "Water",
    "Clouds and Shadows"
  ),
  short = c(
    "No info",
    "Urban",
    "Industrial",
    "Construction",
    "Artificial vegetation",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation",
    "Orchards",
    "Forests",
    "Herbaceous",
    "Open spaces",
    "Wetlands",
    "Water",
    "Clouds"
  ),
  color = c(
    rgb(34, 31, 32, maxColorValue = 255),
    rgb(204, 102, 92, maxColorValue = 255),
    rgb(209, 153, 98, maxColorValue = 255),
    rgb(218, 207, 106, maxColorValue = 255),
    rgb(183, 216, 106, maxColorValue = 255),
    rgb(143, 215, 105, maxColorValue = 255),
    rgb(140, 193, 130, maxColorValue = 255),
    rgb(111, 174, 98, maxColorValue = 255),
    rgb(219, 245, 215, maxColorValue = 255),
    rgb(186, 224, 180, maxColorValue = 255),
    rgb(55, 125, 34, maxColorValue = 255),
    rgb(110, 174, 167, maxColorValue = 255),
    rgb(145, 95, 38, maxColorValue = 255),
    rgb(102, 155, 214, maxColorValue = 255),
    rgb(34, 102, 246, maxColorValue = 255),
    rgb(34, 31, 32, maxColorValue = 255)
  )
)

urban_atlas_dfc22_colormap <- c(
  "0" = rgb(34, 31, 32, maxColorValue = 255),
  "1" = rgb(204, 102, 92, maxColorValue = 255),
  "2" = rgb(209, 153, 98, maxColorValue = 255),
  "3" = rgb(218, 207, 106, maxColorValue = 255),
  "4" = rgb(183, 216, 106, maxColorValue = 255),
  "5" = rgb(143, 215, 105, maxColorValue = 255),
  "6" = rgb(140, 193, 130, maxColorValue = 255),
  "7" = rgb(111, 174, 98, maxColorValue = 255),
  "8" = rgb(219, 245, 215, maxColorValue = 255),
  "9" = rgb(186, 224, 180, maxColorValue = 255),
  "10" = rgb(55, 125, 34, maxColorValue = 255),
  "11" = rgb(110, 174, 167, maxColorValue = 255),
  "12" = rgb(145, 95, 38, maxColorValue = 255),
  "13" = rgb(102, 155, 214, maxColorValue = 255),
  "14" = rgb(34, 102, 246, maxColorValue = 255),
  "15" = rgb(34, 31, 32, maxColorValue = 255)
)
