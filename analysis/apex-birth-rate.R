# load libraries

library(pacman)
p_load(dplyr, tidyr, ggplot2, viridis, cowplot, gridExtra, lme4, RColorBrewer)


# get sim data

data <- read.csv("output/experiments/results/Experiment-5_results.csv")

colnames(data)
head(data)

# representative time series

l <- unique(data$a_breed)[c(0, 10, 20, 30, 40, 50)]

reps <- sample(unique(data$rep_id), 5)

bold <- sample(unique(data$rep_id), 1)

# prey

plot <- data %>%
    filter(a_breed %in% l) %>%
    filter(rep_id %in% reps) %>%
    ggplot(aes(x = step, y = Prey, col = as.factor(a_breed))) +
    geom_line(alpha = 0.25) +
    geom_line(data = data %>% filter(rep_id == bold & a_breed %in% l), alpha = 1, linewidth = 1) +
    scale_color_viridis(discrete = TRUE) +
    theme_bw() +
    labs(
        x = "Time",
        y = "Prey",
        color = "Apex predator breeding rate",
    ) +
    theme(
        legend.position = "none",
        text = element_text(size = 20)
    ) +
    facet_wrap(~a_breed)

ggsave(plot, filename = "output/experiments/plots/prey-time-series-a_breed.png", width = 10, height = 10, dpi = 300)

# Apex

plot <- data %>%
    filter(a_breed %in% l) %>%
    filter(rep_id %in% reps) %>%
    ggplot(aes(x = step, y = Apex, col = as.factor(a_breed))) +
    geom_line(alpha = 0.25) +
    geom_line(data = data %>% filter(rep_id == bold & a_breed %in% l), alpha = 1, linewidth = 1) +
    scale_color_viridis(discrete = TRUE) +
    theme_bw() +
    labs(
        x = "Time",
        y = "Apex predator",
        color = "Apex predator breeding rate"
    ) +
    theme(
        legend.position = "none",
        text = element_text(size = 20)
    ) +
    facet_wrap(~a_breed)

ggsave(plot, filename = "output/experiments/plots/apex-time-serie-a_breed.png", width = 10, height = 10, dpi = 300)

# meso predator

plot <- data %>%
    filter(a_breed %in% l) %>%
    filter(rep_id %in% reps) %>%
    ggplot(aes(x = step, y = Predator, col = as.factor(a_breed))) +
    geom_line(alpha = 0.25) +
    geom_line(data = data %>% filter(rep_id == bold & a_breed %in% l), alpha = 1, linewidth = 1) +
    scale_color_viridis(discrete = TRUE) +
    theme_bw() +
    labs(
        x = "Time",
        y = "Meso predator",
        color = "Apex predator breeding rate"
    ) +
    theme(
        legend.position = "none",
        text = element_text(size = 20)
    ) +
    facet_wrap(~a_breed)

ggsave(plot, filename = "output/experiments/plots/predator-time-serie-a_breed.png", width = 10, height = 10, dpi = 300)

# discard first 400 steps

data <- data %>%
    filter(step > 400)

# equilibrium prey as function of apex predator breeding rate

plot <- data %>%
    ggplot(aes(x = a_breed, y = Prey)) +
    geom_jitter(alpha = 0.05, color = brewer.pal(9, "Set1")[2]) +
    theme_bw() +
    labs(
        x = "Apex predator breeding rate",
        y = "Equilibrium prey"
    ) +
    theme(
        legend.position = "top",
        text = element_text(size = 20)
    )

ggsave(plot, filename = "output/experiments/plots/prey-a_breed.png", width = 10, height = 10, dpi = 300)

# equilibrium meso predator as function of Apex predator breeding rate",
plot <- data %>%
    ggplot(aes(x = a_breed, y = Predator)) +
    geom_jitter(alpha = 0.05, color = brewer.pal(9, "Set1")[1]) + # brewer.pal(9, "Set1")[2]) +
    theme_bw() +
    labs(
        x = "Apex predator breeding rate",
        y = "Equilibrium meso predator"
    ) +
    theme(
        legend.position = "top",
        text = element_text(size = 20)
    )

ggsave(plot, filename = "output/experiments/plots/predator-a_breed.png", width = 10, height = 10, dpi = 300)

# equilibrium apex predator as function of Apex predator breeding rate

plot <- data %>%
    ggplot(aes(x = a_breed, y = Apex)) +
    geom_jitter(alpha = 0.05, color = brewer.pal(9, "Set1")[3]) +
    theme_bw() +
    labs(
        x = "Apex predator breeding rate",
        y = "Equilibrium apex predator"
    ) +
    theme(
        legend.position = "top",
        text = element_text(size = 20)
    )

ggsave(plot, filename = "output/experiments/plots/apex-a_breed.png", width = 10, height = 10, dpi = 300)
