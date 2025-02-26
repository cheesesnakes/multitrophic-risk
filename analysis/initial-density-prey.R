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

predators <- 500
super <- 0
apex <- 0

prey_initial <- initial_densities %>%
    filter(Predator == predators & Super == super & Apex == apex) %>%
    select(sample_id, Prey) %>%
    distinct() %>%
    rename(prey_initial = Prey)

prey_initial

summary(prey_initial)

data <- filter(data, sample_id %in% prey_initial$sample_id)

summary(data)

# representative time series

s <- unique(data$sample_id)[c(1, 5, 10)]
r <- 5

time_series_bold <- data %>%
    filter(sample_id %in% prey_initial$sample_id, rep_id == r) %>%
    select(sample_id, step, Prey, Predator) %>%
    left_join(prey_initial, by = "sample_id")

time_series <- data %>%
    filter(sample_id %in% prey_initial$sample_id) %>%
    select(sample_id, step, Prey, Predator) %>%
    left_join(prey_initial, by = "sample_id")

ggplot(time_series, aes(x = step, y = Prey)) +
    geom_line(alpha = 0.25) +
    geom_line(data = time_series_bold, aes(x = step, y = Prey), color = "red") +
    geom_hline(yintercept = 0, linetype = "dashed") +
    facet_wrap(~prey_initial, scales = "free_y") +
    theme_bw()

ggplot(time_series, aes(x = step, y = Predator)) +
    geom_line(alpha = 0.25) +
    geom_line(data = time_series_bold, aes(x = step, y = Predator), color = "red") +
    geom_hline(yintercept = 0, linetype = "dashed") +
    facet_wrap(~prey_initial, scales = "free_y") +
    theme_bw()

# osciallation diagram

ggplot(time_series, aes(x = Prey, y = Predator)) +
    geom_point(alpha = 0.25) +
    geom_point(data = time_series_bold, aes(x = Prey, y = Predator), color = "red", alpha = 0.25) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_vline(xintercept = 0, linetype = "dashed") +
    facet_wrap(~prey_initial, scales = "free_y") +
    theme_bw()

# bifurcation plots

bifurcation <- data %>%
    filter(step > 400) %>%
    select(sample_id, Prey, Predator) %>%
    left_join(prey_initial, by = "sample_id")

ggplot(bifurcation, aes(x = prey_initial, y = Prey)) +
    geom_jitter(alpha = 0.25) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_vline(xintercept = 0, linetype = "dashed") +
    theme_bw()
