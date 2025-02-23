# load libraries

library(pacman)
p_load(dplyr, tidyr, ggplot2, viridis, cowplot, gridExtra, lme4, RColorBrewer)

# loop through models, skip if file does not exist

experiment_1 <- read.csv("output/experiments/results/Experiment-1_results.csv")
experiment_2 <- read.csv("output/experiments/results/Experiment-2_results.csv")
experiment_3 <- read.csv("output/experiments/results/Experiment-3_results.csv")
experiment_4 <- read.csv("output/experiments/results/Experiment-4_results.csv")

# Experiment 1

n = length(experiment_1$step) / 2

experiment_1$treatment <- c(rep("apex", n), rep("super", n))

# scores

experiment_1 <- experiment_1 %>%
    mutate(
        outcome = ifelse(Prey > 0 & Predator > 0 & step == max(step), "Coexistance",
        ifelse(Prey > Predator, "Prey Only","Extinction"))
    ) %>%
    mutate(score = ifelse(outcome == "Extinction", -1,
        ifelse(outcome == "Prey Only", 0, 1)
    ))

# Number of simulations that ran to completion

print("Number of simulations that ran to completion")
max(experiment_1$step)
print(sum(experiment_1$step == 1999))

# plot N_predator vs N_prey

experiment_1 %>%
    mutate(treatment = factor(treatment, levels = c("apex", "super"))) %>%
    complete(treatment, s_breed, f_breed, fill = list(NA)) %>%
    ggplot(aes(x = Predator, y = Prey, col = f_breed / s_breed)) +
    geom_point(size = 3) +
    labs(
        x = "Predator",
        y = "Prey",
        color = "Prey:Predator breeding rate"
    ) +
    #    scale_y_continuous(limit = c(0,2500))+
    #    scale_x_continuous(limit = c(0,2500))+
    scale_color_viridis() +
    scale_y_log10() +
    scale_x_log10() +
    theme_bw() +
    theme(
        text = element_text(size = 20),
        legend.position = "top"
    ) +
    facet_wrap(~treatment)

ggsave("output/experiments/plots/experiment_1_predator-prey.png", width = 12, height = 7)

# plot mean score by treatment for s_breed vs f_breed

experiment_1_mean <- experiment_1 %>%
    group_by(treatment, s_breed, f_breed) %>%
    summarise(mean_score = mean(score)) %>%
    ungroup() %>%
    mutate(phase = ifelse(mean_score > 0, "Coexistence",
        ifelse(mean_score <= 0 & mean_score >= -0.5, "Prey Only", "Extinction")
    ))

ggplot(experiment_1_mean, aes(x = s_breed, y = f_breed, fill = mean_score)) +
    geom_tile() +
    theme_bw() +
    theme(
        text = element_text(size = 20),
        legend.position = "top"
    ) +
    scale_fill_viridis(
        limits = c(-1, 1), breaks = seq(-1, 1, 0.5),
        guide = guide_colorbar(barwidth = 10, barheight = 2)
    ) +
    scale_x_continuous(breaks = seq(0, 1, 0.1), expand = c(0, 0)) +
    scale_y_continuous(breaks = seq(0, 1, 0.1), expand = c(0, 0)) +
    labs(
        x = "Predator breeding rate",
        y = "Prey breeding rate"
    ) +
    facet_wrap(~treatment)

ggsave("output/experiments/plots/experiment_1_score-plot.png", width = 12, height = 7)

# Phase plot for each treatment

ggplot(experiment_1_mean, aes(x = s_breed, y = f_breed, fill = phase)) +
    geom_tile() +
    theme_bw() +
    theme(
        text = element_text(size = 20),
        legend.position = "top"
    ) +
    scale_fill_brewer(palette = "Set1") +
    scale_x_continuous(breaks = seq(0, 1, 0.1), expand = c(0, 0)) +
    scale_y_continuous(breaks = seq(0, 1, 0.1), expand = c(0, 0)) +
    labs(
        x = "Predator breeding rate",
        y = "Prey breeding rate"
    ) +
    facet_wrap(~treatment)

ggsave("output/experiments/plots/experiment_1_phase-plot.png", width = 12, height = 7)

# get coexitance parameters

experiment_1_coexitance <- experiment_1 %>%
    filter(step == 1999) %>%
    filter(outcome == "Coexistance") %>%
    group_by(treatment, s_breed, f_breed) %>%
    summarise(
        n = n(),
        p = n / 10
    ) %>%
    ungroup() %>%
    # get mean coexitance parameters
    group_by(treatment) %>%
    summarise(
        mean_p = mean(p),
        sd_p = sd(p),
        max_p = max(p),
        min_p = min(p),
        mean_s_breed = mean(s_breed),
        max_s_breed = max(s_breed),
        min_s_breed = min(s_breed),
        mean_f_breed = mean(f_breed),
        max_f_breed = max(f_breed),
        min_f_breed = min(f_breed)
    ) %>%
    ungroup() %>%
    complete(treatment, fill = list(mean_p = 0, sd_p = 0, max_p = 0, min_p = 0, mean_s_breed = 0, max_s_breed = 0, min_s_breed = 0, mean_f_breed = 0, max_f_breed = 0, min_f_breed = 0)) %>%
    pivot_longer(
        cols = c(mean_p, sd_p, max_p, min_p, mean_s_breed, max_s_breed, min_s_breed, mean_f_breed, max_f_breed, min_f_breed),
        names_to = "parameter",
        values_to = "value"
    ) %>%
    pivot_wider(
        names_from = "treatment",
        values_from = "value"
    )

print("Coexitance parameters")
print(experiment_1_coexitance)

# plot outcome by treatment

experiment_1_outcome <- experiment_1 %>%
    group_by(treatment, s_breed, f_breed, outcome) %>%
    summarise(
        n = n(),
        p = n / 10
    ) %>%
    ungroup() %>%
    select(treatment, s_breed, f_breed, outcome, p) %>%
    complete(treatment, s_breed, f_breed, outcome, fill = list(p = 0)) %>%
    # set order of outcome
    mutate(outcome = factor(outcome, levels = c("Prey Only", "Coexistance", "Extinction")))

ggplot(experiment_1_outcome, aes(x = s_breed, y = f_breed, fill = p)) +
    geom_tile() +
    theme_bw() +
    theme(
        text = element_text(size = 25),
        legend.position = "top"
    ) +
    # make scale 0-1
    scale_fill_viridis(
        limits = c(0, 1), breaks = seq(0, 1, 0.5),
        guide = guide_colorbar(barwidth = 10, barheight = 2)
    ) +
    scale_x_continuous(breaks = seq(0, 1, 0.2), expand = c(0, 0)) +
    scale_y_continuous(breaks = seq(0, 1, 0.2), expand = c(0, 0)) +
    labs(
        x = "Predator breeding rate",
        y = "Prey breeding rate"
    ) +
    facet_grid(treatment ~ outcome)

ggsave("output/experiments/plots/experiment_1_outcome.png", width = 18, height = 14)

# Experiment 2

n = length(experiment_2$step) / 2

experiment_2$lethality <- c(rep("Non-lethal", n), rep("Lethal", n))

experiment_2$target <- c(rep("Prey", n/3), rep("Predator", n/3), rep("Both", n/3))

# scores

experiment_2 <- experiment_2 %>%
    mutate(outcome = ifelse(Prey > 0 & Predator > 0 & step == max(step), "Coexistance",
        ifelse(Prey > Predator, "Prey Only","Extinction"))) %>%
    # score
    mutate(score = ifelse(outcome == "Extinction", -1,
        ifelse(outcome == "Prey Only", 0, 0)
    ))

# Number of simulations that ran to completion

print("Number of simulations that ran to completion")
print(sum(experiment_2$step == 1999))

# plot N_pred vs N_prey

experiment_2 %>%
    ggplot(aes(x = Predator, y = Prey, col = f_breed / s_breed)) +
    geom_point(size = 3) +
    labs(
        x = "Predator",
        y = "Prey",
        color = "Prey:Predator breeding rate"
    ) +
    scale_color_viridis() +
    scale_y_log10() +
    scale_x_log10() +
    theme_bw() +
    theme(
        text = element_text(size = 20),
        legend.position = "top"
    ) +
    facet_grid(lethality ~ target)

ggsave("output/experiments/plots/experiment_2_predator-prey.png", width = 18, height = 14)

# plot mean score by treatment for s_breed vs f_breed

experiment_2_mean <- experiment_2 %>%
    group_by(lethality, target, s_breed, f_breed) %>%
    summarise(mean_score = mean(score))

ggplot(experiment_2_mean, aes(x = s_breed, y = f_breed, fill = mean_score)) +
    geom_tile(col = "black") +
    theme_bw() +
    theme(
        text = element_text(size = 20),
        legend.position = "top"
    ) +
    scale_fill_viridis(
        limits = c(-1, 1), breaks = seq(-1, 1, 0.5),
        guide = guide_colorbar(barwidth = 10, barheight = 2)
    ) +
    scale_x_continuous(breaks = seq(0, 1, 0.1)) +
    scale_y_continuous(breaks = seq(0, 1, 0.1)) +
    labs(
        x = "s_breed",
        y = "f_breed"
    ) +
    facet_grid(lethality ~ target)

ggsave("output/experiments/plots/experiment_2_phase-plot.png", width = 18, height = 14)

# plot outcome by treatment

experiment_2_outcome <- experiment_2 %>%
    group_by(lethality, target, s_breed, f_breed, outcome) %>%
    summarise(
        n = n(),
        p = n / 10
    ) %>%
    ungroup() %>%
    select(lethality, target, s_breed, f_breed, outcome, p) %>%
    complete(lethality, target, s_breed, f_breed, outcome, fill = list(p = 0)) %>%
    # set order of outcome
    mutate(outcome = factor(outcome, levels = c("Prey Only", "Coexistance", "Extinction")))

experiment_2_lethal_outcome <- experiment_2_outcome %>%
    filter(lethality == "Lethal")

ggplot(experiment_2_lethal_outcome, aes(x = s_breed, y = f_breed, fill = p)) +
    geom_tile(col = "black") +
    theme_bw() +
    theme(
        text = element_text(size = 20),
        legend.position = "top"
    ) +
    # make scale 0-1
    scale_fill_viridis(
        limits = c(0, 1), breaks = seq(0, 1, 0.5),
        guide = guide_colorbar(barwidth = 10, barheight = 2)
    ) +
    scale_x_continuous(breaks = seq(0, 1, 0.1), expand = c(0, 0)) +
    scale_y_continuous(breaks = seq(0, 1, 0.1), expand = c(0, 0)) +
    labs(
        x = "Predator breeding rate",
        y = "Prey breeding rate"
    ) +
    facet_grid(target ~ outcome)

ggsave("output/experiments/plots/experiment_2_lethal_outcome.png", width = 18, height = 18)

experiment_2_non_lethal_outcome <- experiment_2_outcome %>%
    filter(lethality == "Non-lethal")

ggplot(experiment_2_non_lethal_outcome, aes(x = s_breed, y = f_breed, fill = p)) +
    geom_tile(col = "black") +
    theme_bw() +
    theme(
        text = element_text(size = 20),
        legend.position = "top"
    ) +
    # make scale 0-1
    scale_fill_viridis(
        limits = c(0, 1), breaks = seq(0, 1, 0.5),
        guide = guide_colorbar(barwidth = 10, barheight = 2)
    ) +
    scale_x_continuous(breaks = seq(0, 1, 0.1), expand = c(0, 0)) +
    scale_y_continuous(breaks = seq(0, 1, 0.1), expand = c(0, 0)) +
    labs(
        x = "Predator breeding rate",
        y = "Prey breeding rate"
    ) +
    facet_grid(target ~ outcome)

ggsave("output/experiments/plots/experiment_2_non_lethal_outcome.png", width = 18, height = 18)

# get coexitance parameters

experiment_2_coexitance <- experiment_2 %>%
    filter(step == 1999) %>%
    filter(outcome == "Coexistance") %>%
    group_by(lethality, target, s_breed, f_breed) %>%
    summarise(
        n = n(),
        p = n / 10
    ) %>%
    ungroup() %>%
    # get mean coexitance parameters
    group_by(lethality, target) %>%
    summarise(
        mean_p = mean(p),
        sd_p = sd(p),
        max_p = max(p),
        min_p = min(p),
        mean_s_breed = mean(s_breed),
        max_s_breed = max(s_breed),
        min_s_breed = min(s_breed),
        mean_f_breed = mean(f_breed),
        max_f_breed = max(f_breed),
        min_f_breed = min(f_breed)
    ) %>%
    ungroup() %>%
    complete(lethality, target, fill = list(mean_p = 0, sd_p = 0, max_p = 0, min_p = 0, mean_s_breed = 0, max_s_breed = 0, min_s_breed = 0, mean_f_breed = 0, max_f_breed = 0, min_f_breed = 0)) %>%
    pivot_longer(
        cols = c(mean_p, sd_p, max_p, min_p, mean_s_breed, max_s_breed, min_s_breed, mean_f_breed, max_f_breed, min_f_breed),
        names_to = "parameter",
        values_to = "value"
    ) %>%
    pivot_wider(
        names_from = "target",
        values_from = "value"
    )

print("Coexistance parameters")

print(experiment_2_coexitance)

# parameter for coexitance

experiment_2 %>%
    filter(step == 1999) %>%
    filter(outcome == "Coexistance") %>%
    group_by(lethality, target, s_breed, f_breed) %>%
    summarise(
        n = n(),
        p = n / 10
    ) %>%
    ungroup() %>%
    filter(p == max(p))

# Experiment 3

n = length(experiment_3$step)

experiment_3$predator_info <- c(rep("Predator Informed", n/4), rep("Predator Naive", n/4), rep("Predator Informed", n/4), rep("Predator Naive", n/4))
experiment_3$prey_info <- c(rep("Prey Informed", n/2), rep("Prey Naive", n/2))

# scores

experiment_3 <- experiment_3 %>%
    mutate(outcome = ifelse(Prey > 0 & Predator > 0 & step == max(step), "Coexistance",
        ifelse(Prey > Predator, "Prey Only","Extinction")
        ))%>%
    # score
    mutate(score = ifelse(outcome == "Extinction", -1,
        ifelse(outcome == "Prey Only", 0, 0)
    ))

# Number of simulations that ran to completion

print("Number of simulations that ran to completion")
print(sum(experiment_3$step == 1999))

# plot N_pred vs N_prey

experiment_3 %>%
    ggplot(aes(x = Predator, y = Prey, col = f_breed / s_breed)) +
    geom_point(size = 3) +
    labs(
        x = "Predator",
        y = "Prey",
        color = "Prey:Predator breeding rate"
    ) +
    scale_color_viridis() +
    scale_y_log10() +
    scale_x_log10() +
    theme_bw() +
    theme(
        text = element_text(size = 20),
        legend.position = "top"
    ) +
    facet_grid(predator_info ~ prey_info)

ggsave("output/experiments/plots/experiment_3_predator-prey.png", width = 12, height = 14)

# plot mean score by treatment for s_breed vs f_breed

experiment_3_mean <- experiment_3 %>%
    group_by(predator_info, prey_info, s_breed, f_breed) %>%
    summarise(mean_score = mean(score))

ggplot(experiment_3_mean, aes(x = s_breed, y = f_breed, fill = mean_score)) +
    geom_tile(col = "black") +
    theme_bw() +
    theme(
        text = element_text(size = 20),
        legend.position = "top"
    ) +
    scale_fill_viridis(
        limits = c(-1, 1), breaks = seq(-1, 1, 0.5),
        guide = guide_colorbar(barwidth = 10, barheight = 2)
    ) +
    scale_x_continuous(breaks = seq(0, 1, 0.1)) +
    scale_y_continuous(breaks = seq(0, 1, 0.1)) +
    labs(
        x = "s_breed",
        y = "f_breed"
    ) +
    facet_grid(predator_info ~ prey_info)

ggsave("output/experiments/plots/experiment_3_phase-plot.png", width = 12, height = 14)

# plot outcome by treatment

experiment_3_outcome <- experiment_3 %>%
    group_by(predator_info, prey_info, s_breed, f_breed, outcome) %>%
    summarise(
        n = n(),
        p = n / 10
    ) %>%
    ungroup() %>%
    select(predator_info, prey_info, s_breed, f_breed, outcome, p) %>%
    complete(predator_info, prey_info, s_breed, f_breed, outcome, fill = list(p = 0)) %>%
    # set order of outcome
    mutate(outcome = factor(outcome, levels = c("Prey Only", "Coexistance", "Extinction"))) %>%
    mutate(treatment = ifelse(predator_info == "Predator Informed" & prey_info == "Prey Informed", "Both Informed",
        ifelse(predator_info == "Predator Informed" & prey_info == "Prey Naive", "Predator Informed",
            ifelse(predator_info == "Predator Naive" & prey_info == "Prey Informed", "Prey Informed", "Both Naive")
        )
    )) %>%
    mutate(treatment = factor(treatment, levels = c("Both Informed", "Predator Informed", "Prey Informed", "Both Naive")))

ggplot(experiment_3_outcome, aes(x = s_breed, y = f_breed, fill = p)) +
    geom_tile(col = "black") +
    theme_bw() +
    theme(
        text = element_text(size = 20),
        legend.position = "top"
    ) +
    # make scale 0-1
    scale_fill_viridis(
        limits = c(0, 1), breaks = seq(0, 1, 0.5),
        guide = guide_colorbar(barwidth = 10, barheight = 2)
    ) +
    scale_x_continuous(breaks = seq(0, 1, 0.1), expand = c(0, 0)) +
    scale_y_continuous(breaks = seq(0, 1, 0.1), expand = c(0, 0)) +
    labs(
        x = "Predator breeding rate",
        y = "Prey breeding rate"
    ) +
    facet_grid(treatment ~ outcome)

ggsave("output/experiments/plots/experiment_3_outcome.png", width = 18, height = 24)

# get coexitance parameters

experiment_3_coexitance <- experiment_3 %>%
    filter(step == 1999) %>%
    filter(outcome == "Coexistance") %>%
    group_by(predator_info, prey_info, s_breed, f_breed) %>%
    summarise(
        n = n(),
        p = n / 10
    ) %>%
    ungroup() %>%
    # get mean coexitance parameters
    mutate(treatment = ifelse(predator_info == "Predator Informed" & prey_info == "Prey Informed", "Both Informed",
        ifelse(predator_info == "Predator Informed" & prey_info == "Prey Naive", "Predator Informed",
            ifelse(predator_info == "Predator Naive" & prey_info == "Prey Informed", "Prey Informed", "Both Naive")
        )
    )) %>%
    mutate(treatment = factor(treatment, levels = c("Both Informed", "Predator Informed", "Prey Informed", "Both Naive"))) %>%
    group_by(treatment) %>%
    summarise(
        mean_p = mean(p),
        sd_p = sd(p),
        max_p = max(p),
        min_p = min(p),
        mean_s_breed = mean(s_breed),
        max_s_breed = max(s_breed),
        min_s_breed = min(s_breed),
        mean_f_breed = mean(f_breed),
        max_f_breed = max(f_breed),
        min_f_breed = min(f_breed)
    ) %>%
    ungroup() %>%
    complete(treatment, fill = list(mean_p = 0, sd_p = 0, max_p = 0, min_p = 0, mean_s_breed = 0, max_s_breed = 0, min_s_breed = 0, mean_f_breed = 0, max_f_breed = 0, min_f_breed = 0)) %>%
    pivot_longer(
        cols = c(mean_p, sd_p, max_p, min_p, mean_s_breed, max_s_breed, min_s_breed, mean_f_breed, max_f_breed, min_f_breed),
        names_to = "parameter",
        values_to = "value"
    ) %>%
    pivot_wider(
        names_from = "treatment",
        values_from = "value"
    )

print("Coexistance parameters")

print(experiment_3_coexitance)

# Experiment 4

# scores

experiment_4 <- experiment_4 %>%
    mutate(outcome = ifelse(Prey > 0 & Predator > 0 & step == max(step), "Coexistance",
        ifelse(Prey > Predator, "Prey Only","Extinction"))) %>%
    # score
    mutate(score = ifelse(outcome == "Extinction", -1,
        ifelse(outcome == "Prey Only", 0, 0)
    ))

# Number of simulations that ran to completion

print("Number of simulations that ran to completion")
print(sum(experiment_4$step == 1999))

# plot N_pred vs N_prey

experiment_4 %>%
    ggplot(aes(x = Predator, y = Prey, col = s_max / a_max)) +
    geom_point(size = 3) +
    labs(
        x = "Predator",
        y = "Prey",
        color = "Prey:Predator breeding rate"
    ) +
    scale_color_viridis() +
    scale_y_log10() +
    scale_x_log10() +
    theme_bw() +
    theme(
        text = element_text(size = 20),
        legend.position = "top"
    )

ggsave("output/experiments/plots/experiment_4_predator-prey.png", width = 7, height = 7)

# plot mean score by treatment for a_max vs s_max

experiment_4_mean <- experiment_4 %>%
    group_by(s_max, a_max) %>%
    summarise(mean_score = mean(score)) %>%
    ungroup() %>%
    mutate(phase = ifelse(mean_score > 0, "Coexistence",
        ifelse(mean_score == 0, "Prey Only", "Extinction")
    ))

ggplot(experiment_4_mean, aes(x = s_max, y = a_max, fill = mean_score)) +
    geom_tile(col = "black") +
    theme_bw() +
    theme(
        text = element_text(size = 20),
        legend.position = "top"
    ) +
    scale_fill_viridis(
        limits = c(-1, 1), breaks = seq(-1, 1, 0.5),
        guide = guide_colorbar(barwidth = 10, barheight = 2)
    ) +
    scale_x_continuous(breaks = seq(0, 100, 10)) +
    scale_y_continuous(breaks = seq(0, 100, 10)) +
    labs(
        x = "s_max",
        y = "a_max"
    )

ggsave("output/experiments/plots/experiment_4_score-plot.png", width = 7, height = 7)

# Phase plot for each treatment

ggplot(experiment_4_mean, aes(x = s_max, y = a_max, fill = phase)) +
    geom_tile() +
    theme_bw() +
    theme(
        text = element_text(size = 20),
        legend.position = "top"
    ) +
    scale_fill_brewer(palette = "Set1") +
    scale_x_continuous(breaks = seq(0, 100, 10)) +
    scale_y_continuous(breaks = seq(0, 100, 10)) +
    labs(
        x = "s_max",
        y = "a_max"
    )

ggsave("output/experiments/plots/experiment_4_phase-plot.png", width = 7, height = 7)



# plot outcome by treatment

experiment_4_outcome <- experiment_4 %>%
    group_by(s_max, a_max, outcome) %>%
    summarise(
        n = n(),
        p = n / 10
    ) %>%
    ungroup() %>%
    select(s_max, a_max, outcome, p) %>%
    complete(s_max, a_max, outcome, fill = list(p = 0)) %>%
    # set order of outcome
    mutate(outcome = factor(outcome, levels = c("Prey Only", "Coexistance", "Extinction"))) %>%
    mutate(p = ifelse(p > 1, 1, p))


ggplot(experiment_4_outcome, aes(x = s_max, y = a_max, fill = p)) +
    geom_tile(col = "black") +
    theme_bw() +
    theme(
        text = element_text(size = 20),
        legend.position = "top"
    ) +
    # make scale 0-1
    scale_fill_viridis(
        limits = c(0, 1), breaks = seq(0, 1, 0.5),
        guide = guide_colorbar(barwidth = 10, barheight = 2)
    ) +
    scale_x_continuous(breaks = seq(0, 100, 10), expand = c(0, 0)) +
    scale_y_continuous(breaks = seq(0, 100, 10), expand = c(0, 0)) +
    labs(
        x = "Mesopredator Starting max",
        y = "Apex-predator Starting max"
    ) +
    facet_wrap(~outcome)

ggsave("output/experiments/plots/experiment_4_outcome.png", width = 18, height = 7)
