# load libraries

library(pacman)
p_load(dplyr, tidyr, ggplot2, viridis, cowplot, gridExtra, lme4, RColorBrewer)


# get sim data

data <- read.csv("output/experiments/results/Experiment-7_results.csv")

colnames(data)
head(data)
summary(data)

# representative time series

l <- unique(data$s_lethality)[c(13,26,38,50)]

reps <- sample(unique(data$rep_id), 5)

bold <- sample(unique(data$rep_id), 1)

# prey

plot <- data %>%
    filter(s_lethality %in% l) %>%
    filter(rep_id %in% reps) %>%
    ggplot(aes(x = step, y = Prey, col = as.factor(s_lethality))) +
    geom_line(alpha = 0.25) +
    geom_line(data = data %>% filter(rep_id == bold & s_lethality %in% l), alpha = 1, linewidth = 1) +
    scale_color_viridis(discrete = TRUE) +
    theme_bw() +
    labs(
        x = "Time",
        y = "Prey",
        color = "Meso predator lethality"
    ) +
    theme(
        legend.position = "none",
        text = element_text(size = 20)
    ) +
    facet_wrap(~s_lethality)

ggsave(plot, filename = "output/experiments/plots/prey-time-series_lethality.png", width = 10, height = 10, dpi = 300)

# meso predator

plot <- data %>%
    filter(s_lethality %in% l) %>%
    filter(rep_id %in% reps) %>%
    ggplot(aes(x = step, y = Predator, col = as.factor(s_lethality))) +
    geom_line(alpha = 0.25) +
    geom_line(data = data %>% filter(rep_id == bold & s_lethality %in% l), alpha = 1, linewidth = 1) +
    scale_color_viridis(discrete = TRUE) +
    theme_bw() +
    labs(
        x = "Time",
        y = "Meso predator",
        color = "Meso predator lethality"
    ) +
    theme(
        legend.position = "none",
        text = element_text(size = 20)
    ) +
    facet_wrap(~s_lethality)

ggsave(plot, filename = "output/experiments/plots/predator-time-series_lethality.png", width = 10, height = 10, dpi = 300)

# discard first 400 steps

data <- data %>%
    filter(step > 400)

# equilibrium prey as function of meso predator lethality

plot <- data %>%
    ggplot(aes(x = s_lethality, y = Prey)) +
    geom_jitter(alpha = 0.05, color = brewer.pal(9, "Set1")[2]) +
    theme_bw() +
    labs(x = "Meso predator lethality",
         y = "Equilibrium prey") +
    theme(legend.position = "top",
          text = element_text(size = 20))

ggsave(plot, filename = "output/experiments/plots/prey-lethality.png", width = 10, height = 10, dpi = 300)

# equilibrium meso predator as function of meso predator lethality

plot <- data %>%
    ggplot(aes(x = s_lethality, y = Predator)) +
    geom_jitter(alpha = 0.05, color = brewer.pal(9, "Set1")[1]) + #brewer.pal(9, "Set1")[2]) +
    theme_bw() +
    labs(x = "Meso predator lethality",
         y = "Equilibrium meso predator") +
    theme(legend.position = "top",
          text = element_text(size = 20))

ggsave(plot, filename = "output/experiments/plots/predator-lethality.png", width = 10, height = 10, dpi = 300)
