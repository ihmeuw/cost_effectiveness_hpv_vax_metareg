# Load libraries
library(data.table)
library(raster)
library(dplyr)
library(ggplot2)
library(sf)
library(rgeos)

# If running interactively, must set working directory to the directory containing this file.
paths <- readLines("./paths.py")
paths <- grep("=", paths, value=TRUE)
paths <- gsub("= ?", "= paste0(", paths)
paths <- gsub(" ?\\+", ",", paths)
paths <- paste0(paths, ")")
for(pth in paths){
  eval(parse(text=pth))
}
adj_predictions_path <- paste0(MODEL_RESULTS_DIR, "predictions_with_adj.csv")

shape <- readRDS(paste0(SHAPEFILE))
asf <- st_as_sf(shape)

shape@data$ADM0_CODE <- as.numeric(as.character(shape@data$ADM0_CODE))
asf$ADM0_CODE <- as.numeric(as.character(asf$ADM0_CODE))

data <- fread(adj_predictions_path)
data[, gdp_cat := c("< 0.5"=1, "0.5 - 1"=2, "1 - 3"=3, "3 -  Inf"=4)[GDP_category]]
setnames(data, "location_id", "loc_id")

breaks <- quantile(data[lancet_label != "", adj_ICER], (0:6)/6, type=5)
breaks[1] <- 0
breaks[length(breaks)] <- Inf
breaks[2:(length(breaks)-1)] <- 0.5*round(2*breaks[2:(length(breaks)-1)], -2)
data[, adj_icer_decile := cut(adj_ICER, breaks=breaks, labels = seq(0, 5, length=6))]

shape@data$loc_id <- as.integer(as.character(shape@data$loc_id))
shape@data <- merge(shape@data, data, by = 'loc_id', all.x = TRUE)

##### Making map of the adjusted ICERs

# Subset map to countries where there is ICER data
shape_no_na <-shape[!is.na(shape@data$adj_ICER),]
data_no_na <- st_as_sf(shape_no_na)

# Manual color scale based on deciles of adjusted ICERs
colors <- c('0'='#ffffcc', '1'='#c7e9b4', '2'= '#7fcdbb', '3'='#41b6c4', '4'= '#2c7fb8', '5'='#253494' )

lbls <- paste(breaks[-length(breaks)], breaks[-1], sep=" to ")
lbls <- gsub("^0 to ", "< ", lbls)
lbls[grepl("Inf", lbls)] <- paste0("> ", gsub(" to Inf", "", lbls[grepl("Inf", lbls)]))

# Code to actually make adjusted ICER plot
adj_icer_map <- ggplot() +
  # Map all countries with gray as a canvas
  geom_sf(data = asf, color = 'black', fill = 'grey50', lwd = 0.05) +
  # Map data; "mean" is the variable you want to map
  geom_sf(data = data_no_na, aes(fill = as.character(adj_icer_decile)), lwd = 0) +
  # Map country borders
  geom_sf(data = asf, color = 'black', fill = NA, lwd = 0.05) +
  # Use manual mapping to determine color based on which decile an icer falls into
  scale_fill_manual(
    na.value = 'gray50',
    values = colors,
    name = 'Adjusted ICER\n(2017 US$/DALY averted)',
    labels = lbls) +
  theme_classic() +
  # Zooms in the map
  xlim(extent(data_no_na)[1], extent(data_no_na)[2]) +
  ylim(extent(data_no_na)[3], extent(data_no_na)[4]) +
  # Makes the map pretty
  theme(legend.position = c(0, 0), legend.justification = c(0, 0),
        legend.text=element_text(size=5),
        legend.key.size = unit(0.3, "cm"),
        plot.title = element_text(hjust=0.5),
        plot.margin=unit(c(0, 0, 0, 0), "in")) +
  guides(fill = guide_legend(reverse = FALSE)) +
  coord_sf(datum=NA)

file_path <- paste0(ROOT_DIR, "HPV_vaccines_adj_icer.png")
if(!file.exists(file_path)){
  png(paste0(ROOT_DIR, "HPV_vaccines_adj_icer.png"), width = 8, height = 4, units = 'in', res = 1200)
  print(adj_icer_map)
  dev.off()
}
#######################################################################
colors <- c('1' = "#ffffb2", '2'="#fecc5c", '3' = "#fd8d3c", '4' = "#e31a1c")
lbls <- c('< 0.5x GDP','0.5-0.99x GDP', '1-3x GDP', '> 3x GDP')

#### GDP cutoff map ###
gdp_cutoff_map <- ggplot() +
  # Map all countries with gray as a canvas
  geom_sf(data = asf, color = 'black', fill = 'gray50', lwd = 0.05) +
  # Map data
  geom_sf(data = data_no_na, aes(fill = as.character(gdp_cat)), lwd = 0) +
  # Map country borders
  geom_sf(data = asf, color = 'black', fill = NA, lwd = 0.05) +
  scale_fill_manual(
    na.value = 'gray50',
    values = colors,
    name = 'Predicted ICERs relative to\nGDP per capita thresholds\n(2017 US$/DALY averted)',
    labels = lbls) +
  theme_classic() +
  # Zoom in
  xlim(extent(data_no_na)[1], extent(data_no_na)[2]) +
  ylim(extent(data_no_na)[3], extent(data_no_na)[4]) +
  # Makes the map pretty
  theme(legend.position = c(0, 0), legend.justification = c(0, 0),
        legend.text=element_text(size=5),
        legend.key.size = unit(0.3, "cm"),
        plot.title = element_text(hjust=0.5),
        plot.margin=unit(c(0, 0, 0, 0), 'in')) +
  guides(fill = guide_legend(reverse =FALSE)) +
  coord_sf(datum=NA)

# Save Plot
png(paste0(ROOT_DIR, "HPV_vaccines_GDP_threshold.png"),
    width = 8, height = 4, units = 'in', res = 1200)
print(gdp_cutoff_map)
dev.off()

