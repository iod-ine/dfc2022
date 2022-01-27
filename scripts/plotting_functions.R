plot_tci <- function(file) {
  tci <- rast(file)

  tm_shape(tci) +
    tm_rgb() +
    tm_layout(
      frame = FALSE,
    )
}

plot_dem <- function(file) {
  dem <- rast(file)

  slope <- terrain(dem, "slope", unit = "radians")
  aspect <- terrain(dem, "aspect", unit = "radians")
  shade <- shade(slope, aspect, 40, 300)

  tm_shape(dem) +
    tm_raster(
      style = "cont",
      title = "Elevation",
      palette = terrain.colors(50),
      legend.show = FALSE,
    ) +
    tm_layout(
      frame = FALSE,
    ) +
    tm_shape(shade) +
    tm_raster(
      style = "cont",
      palette = grey(0:10 / 10),
      alpha = 0.4,
      legend.show = FALSE,
    )
}

plot_mask <- function(file) {
  mask <- rast(file)

  tm_shape(mask) +
    tm_raster(
      style = "fixed",
      palette = urban_atlas_dfc22_palette$color,
      breaks = -1:15,
      interval.closure = "right",
      labels = urban_atlas_dfc22_palette$short,
      title = "Class",
      legend.show = FALSE,
    ) +
    tm_layout(
      frame = FALSE,
    )
}

plot_aux <- function(mask_file, title) {
  mask <- rast(mask_file)

  tm_shape(mask) +
    tm_raster(
      style = "fixed",
      palette = urban_atlas_dfc22_palette$color,
      breaks = -1:15,
      interval.closure = "right",
      labels = urban_atlas_dfc22_palette$short,
      title = title,
      legend.show = TRUE,
      legend.is.portrait = TRUE,
    ) +
    tm_layout(
      frame = FALSE,
      legend.only = TRUE,
    ) +
    tm_scale_bar()
}

