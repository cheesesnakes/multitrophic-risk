# load libraries

library(pacman)
p_load(dplyr, tidyr, ggplot2, viridis, cowplot, gridExtra, lme4, RColorBrewer)


# get sim data

data <- read.csv("output/experiments/results/Experiment-4_apex_results.csv")

colnames(data)
head(data)

# representative time series

l <- unique(data$a_max)[c(1,5,20,30,40,50)]

reps <- sample(unique(data$rep_id), 5)

bold <- sample(unique(data$rep_id), 1)

# prey

plot <- data %>%
    filter(a_max %in% l) %>%
    filter(rep_id %in% reps) %>%
    ggplot(aes(x = step, y = Prey, col = as.factor(a_max))) +
    geom_line(alpha = 0.25) +
    geom_line(data = data %>% filter(rep_id == bold & a_max %in% l), alpha = 1, linewidth = 1) +
    scale_color_viridis(discrete = TRUE) +
    theme_bw() +
    labs(
        x = "Time",
        y = "Prey",
        color = "Apex predator handling limit",
    ) +
    theme(
        legend.position = "none",
        text = element_text(size = 20)
    ) +
    facet_wrap(~a_max)

ggsave(plot, filename = "output/experiments/plots/prey-time-series-a_max.png", width = 10, height = 10, dpi = 300)

# Apex

plot <- data %>%
    filter(a_max %in% l) %>%
    filter(rep_id %in% reps) %>%
    ggplot(aes(x = step, y = Apex, col = as.factor(a_max))) +
    geom_line(alpha = 0.25) +
    geom_line(data = data %>% filter(rep_id == bold & a_max %in% l), alpha = 1, linewidth = 1) +
    scale_color_viridis(discrete = TRUE) +
    theme_bw() +
    labs(
        x = "Time",
        y = "Apex predator",
        color = "Apex predator handling limit"
    ) +
    theme(
        legend.position = "none",
        text = element_text(size = 20)
    ) +
    facet_wrap(~a_max)

ggsave(plot, filename = "output/experiments/plots/apex-time-serie-a_max.png", width = 10, height = 10, dpi = 300)

# meso predator

plot <- data %>%
    filter(a_max %in% l) %>%
    filter(rep_id %in% reps) %>%
    ggplot(aes(x = step, y = Predator, col = as.factor(a_max))) +
    geom_line(alpha = 0.25) +
    geom_line(data = data %>% filter(rep_id == bold & a_max %in% l), alpha = 1, linewidth = 1) +
    scale_color_viridis(discrete = TRUE) +
    theme_bw() +
    labs(
        x = "Time",
        y = "Meso predator",
        color = "Apex predator handling limit"
    ) +
    theme(
        legend.position = "none",
        text = element_text(size = 20)
    ) +
    facet_wrap(~a_max)

ggsave(plot, filename = "output/experiments/plots/predator-time-serie-a_max.png", width = 10, height = 10, dpi = 300)

# discard first 400 steps

data <- data %>%
    filter(step > 400)

# equilibrium prey as function of apex predator handling limit

plot <- data %>%
    ggplot(aes(x = a_max, y = Prey)) +
    geom_jitter(alpha = 0.05, color = brewer.pal(9, "Set1")[2]) +
    theme_bw() +
    labs(x = "Apex predator handling limit",
             y = "Equilibrium prey") +
    theme(legend.position = "top",
          text = element_text(size = 20))

ggsave(plot, filename = "output/experiments/plots/prey-a_max.png", width = 10, height = 10, dpi = 300)

# equilibrium meso predator as function of Apex predator handling limit",
plot <- data %>%
    ggplot(aes(x = a_max, y = Predator)) +
    geom_jitter(alpha = 0.05, color = brewer.pal(9, "Set1")[1]) + #brewer.pal(9, "Set1")[2]) +
    theme_bw() +
    labs(x = "Apex predator handling limit",         
    y = "Equilibrium meso predator") +
    theme(legend.position = "top",
          text = element_text(size = 20))

ggsave(plot, filename = "output/experiments/plots/predator-a_max.png", width = 10, height = 10, dpi = 300)

# equilibrium apex predator as function of Apex predator handling limit

plot <- data %>%
    ggplot(aes(x = a_max, y = Apex)) +
    geom_jitter(alpha = 0.05, color = brewer.pal(9, "Set1")[3]) +
    theme_bw() +
    labs(
        x = "Apex predator handling limit",
        y = "Equilibrium apex predator"
    ) +
    theme(
        legend.position = "top",
        text = element_text(size = 20)
    )

ggsave(plot, filename = "output/experiments/plots/apex-a_max.png", width = 10, height = 10, dpi = 300)

# load libraries

library(pacman)
p_load(dplyr, tidyr, ggplot2, viridis, cowplot, gridExtra, lme4, RColorBrewer)


# get sim data

data <- read.csv("output/experiments/results/Experiment-4_lv_results.csv")

colnames(data)
head(data)

# representative time series

l <- unique(data$s_max)[c(1,5,20,30,40,50)]

reps <- sample(unique(data$rep_id), 5)

bold <- sample(unique(data$rep_id), 1)

# prey

plot <- data %>%
    filter(s_max %in% l) %>%
    filter(rep_id %in% reps) %>%
    ggplot(aes(x = step, y = Prey, col = as.factor(s_max))) +
    geom_line(alpha = 0.25) +
    geom_line(data = data %>% filter(rep_id == bold & s_max %in% l), alpha = 1, linewidth = 1) +
    scale_color_viridis(discrete = TRUE) +
    theme_bw() +
    labs(
        x = "Time",
        y = "Prey",
        color = "Meso predator handling limit",
    ) +
    theme(
        legend.position = "none",
        text = element_text(size = 20)
    ) +
    facet_wrap(~s_max)

ggsave(plot, filename = "output/experiments/plots/prey-time-series-s_max.png", width = 10, height = 10, dpi = 300)

# meso predator

plot <- data %>%
    filter(s_max %in% l) %>%
    filter(rep_id %in% reps) %>%
    ggplot(aes(x = step, y = Predator, col = as.factor(s_max))) +
    geom_line(alpha = 0.25) +
    geom_line(data = data %>% filter(rep_id == bold & s_max %in% l), alpha = 1, linewidth = 1) +
    scale_color_viridis(discrete = TRUE) +
    theme_bw() +
    labs(
        x = "Time",
        y = "Meso predator",
        color = "Meso predator handling limit"
    ) +
    theme(
        legend.position = "none",
        text = element_text(size = 20)
    ) +
    facet_wrap(~s_max)

ggsave(plot, filename = "output/experiments/plots/predator-time-serie-s_max.png", width = 10, height = 10, dpi = 300)

# discard first 400 steps

data <- data %>%
    filter(step > 400)

# equilibrium prey as function of Meso predator handling limit

plot <- data %>%
    ggplot(aes(x = s_max, y = Prey)) +
    geom_jitter(alpha = 0.05, color = brewer.pal(9, "Set1")[2]) +
    theme_bw() +
    labs(x = "Meso predator handling limit",
             y = "Equilibrium prey") +
    theme(legend.position = "top",
          text = element_text(size = 20))

ggsave(plot, filename = "output/experiments/plots/prey-s_max.png", width = 10, height = 10, dpi = 300)

# equilibrium meso predator as function of Meso predator handling limit",
plot <- data %>%
    ggplot(aes(x = s_max, y = Predator)) +
    geom_jitter(alpha = 0.05, color = brewer.pal(9, "Set1")[1]) + #brewer.pal(9, "Set1")[2]) +
    theme_bw() +
    labs(x = "Meso predator handling limit",         
    y = "Equilibrium meso predator") +
    theme(legend.position = "top",
          text = element_text(size = 20))

ggsave(plot, filename = "output/experiments/plots/predator-s_max.png", width = 10, height = 10, dpi = 300)
