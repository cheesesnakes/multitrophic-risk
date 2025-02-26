# load libraries

library(pacman)
p_load(dplyr, tidyr, ggplot2, viridis, cowplot, gridExtra, lme4, RColorBrewer)


# get sim data

data <- read.csv("output/experiments/results/Experiment-9_results.csv")

colnames(data)
head(data)

# make sample ids

reps <- 10
t <- 1000

initial_densities <- data %>%
    select(rep_id, sample_id, Prey, Predator, Apex, Super, step) %>%
    filter(step == 0)

sample_n <- (
    length(unique(initial_densities$Prey)) *
        length(unique(initial_densities$Predator)) *
        length(unique(initial_densities$Apex)) *
        length(unique(initial_densities$Super)))

initial_densities$sample_id <- rep(1:sample_n, each = reps)

summary(initial_densities)

data$sample_id <- rep(1:sample_n, each = reps * t)

summary(data)

# varying prey density

prey <- 500
predator <- 500
super <- 0

apex_initial <- initial_densities %>%
    filter(Prey == prey & Predator == predator & Super == super) %>%
    select(sample_id, Apex) %>%
    distinct() %>%
    rename(apex_initial = Apex)

apex_initial

summary(apex_initial)

data <- filter(data, sample_id %in% apex_initial$sample_id)

summary(data)

# representative time series

s <- unique(data$sample_id)[c(1, 5, 10)]
r <- 5

time_series_bold <- data %>%
    filter(sample_id %in% apex_initial$sample_id, rep_id == r) %>%
    select(sample_id, step, Prey, Predator, Apex) %>%
    left_join(apex_initial, by = "sample_id")

time_series <- data %>%
    filter(sample_id %in% apex_initial$sample_id) %>%
    select(sample_id, step, Prey, Predator, Apex) %>%
    left_join(apex_initial, by = "sample_id")

ggplot(time_series, aes(x = step, y = Prey)) +
    geom_line(alpha = 0.25) +
    geom_line(data = time_series_bold, aes(x = step, y = Prey), color = "red") +
    geom_hline(yintercept = 0, linetype = "dashed") +
    facet_wrap(~apex_initial, scales = "free_y") +
    theme_bw()

ggplot(time_series, aes(x = step, y = Predator)) +
    geom_line(alpha = 0.25) +
    geom_line(data = time_series_bold, aes(x = step, y = Predator), color = "red") +
    geom_hline(yintercept = 0, linetype = "dashed") +
    facet_wrap(~apex_initial, scales = "free_y") +
    theme_bw()

ggplot(time_series, aes(x = step, y = Apex)) +
    geom_line(alpha = 0.25) +
    geom_line(data = time_series_bold, aes(x = step, y = Apex), color = "red") +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_vline(xintercept = 0, linetype = "dashed") +
    facet_wrap(~apex_initial) +
    theme_bw()

# oscillation diagram

ggplot(time_series, aes(x = Prey, y = Predator)) +
    geom_point(alpha = 0.25) +
    geom_point(data = time_series_bold, aes(x = Prey, y = Predator), color = "red", alpha = 0.25) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_vline(xintercept = 0, linetype = "dashed") +
    facet_wrap(~apex_initial) +
    theme_bw()

ggplot(time_series, aes(x = Prey, y = Apex)) +
    geom_point(alpha = 0.25) +
    geom_point(data = time_series_bold, aes(x = Prey, y = Apex), color = "red", alpha = 0.25) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_vline(xintercept = 0, linetype = "dashed") +
    facet_wrap(~apex_initial) +
    theme_bw()

ggplot(time_series, aes(x = Predator, y = Apex)) +
    geom_point(alpha = 0.25) +
    geom_point(data = time_series_bold, aes(x = Predator, y = Apex), color = "red", alpha = 0.25) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_vline(xintercept = 0, linetype = "dashed") +
    facet_wrap(~apex_initial) +
    theme_bw()

# bifurcation plots


bifurcation <- data %>%
    filter(step > 400) %>%
    select(sample_id, Prey, Predator, Apex) %>%
    left_join(apex_initial, by = "sample_id")

ggplot(bifurcation, aes(x = apex_initial, y = Prey)) +
    geom_jitter(alpha = 0.25) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_vline(xintercept = 0, linetype = "dashed") +
    theme_bw()

ggplot(bifurcation, aes(x = apex_initial, y = Predator)) +
    geom_jitter(alpha = 0.25) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_vline(xintercept = 0, linetype = "dashed") +
    theme_bw()

ggplot(bifurcation, aes(x = apex_initial, y = Apex)) +
    geom_point(alpha = 0.25) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_vline(xintercept = 0, linetype = "dashed") +
    theme_bw()
