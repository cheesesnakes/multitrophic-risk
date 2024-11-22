
# load libraries

library(pacman)
p_load(dplyr, tidyr, ggplot2, viridis, cowplot, gridExtra, lme4)


# loop through models, skip if file does not exist

experiment_1 = read.csv("output/experiments/results/Experiment-1_results.csv")
experiment_2 = read.csv("output/experiments/results/Experiment-2_results.csv")

# Experiment 1

experiment_1$treatment = c(rep("apex", 4000), rep("super", 4000))

# scores 

experiment_1 = experiment_1%>%
mutate(outcome = ifelse(Prey == 0, "Extinction", 
                        ifelse(Predator == 0, "Prey Only", "Coexitance")))%>%
# score 
mutate(score = ifelse(outcome == "Extinction", -1, 
                      ifelse(outcome == "Prey Only", 0, 0)))

# Number of simulations that ran to completion

print("Number of simulations that ran to completion")
max(experiment_1$step)
print(sum(experiment_1$step == 1999))

# plot N_predator vs N_prey

experiment_1%>%
filter(Predator > 0, Prey > 0, Predator + Prey < 100000)%>%
ggplot(aes(x = Predator, y = Prey, col = f_breed/s_breed))+
    geom_point(size = 3)+
    labs(x = "Predator",
            y = "Prey",
            color = "Prey:Predator breeding rate")+
#    scale_y_continuous(limit = c(0,2500))+
#    scale_x_continuous(limit = c(0,2500))+
    scale_color_viridis()+
    theme_bw()+
    theme(text = element_text(size = 20),
    legend.position = "top")+
    facet_wrap(~treatment)

ggsave("output/experiments/plots/experiment_1_predator-prey.png", width = 10, height = 7)

# plot mean score by treatment for s_breed vs f_breed

experiment_1_mean = experiment_1%>%
    group_by(treatment, s_breed, f_breed)%>%
    summarise(mean_score = mean(score))

ggplot(experiment_1_mean, aes(x = s_breed, y = f_breed, fill = mean_score))+
    geom_tile()+
    theme_bw()+
    theme(text = element_text(size = 20),
    legend.position = "top")+
    scale_fill_viridis(limits = c(-1,1), breaks = seq(-1, 1, 0.5),
    guide = guide_colorbar(barwidth = 10, barheight = 2))+
    scale_x_continuous(breaks = seq(0, 1, 0.1), expand = c(0,0))+
    scale_y_continuous(breaks = seq(0, 1, 0.1), expand = c(0,0))+
    labs(x = "Predator breeding rate",
            y = "Prey breeding rate")+
    facet_wrap(~treatment)

ggsave("output/experiments/plots/experiment_1_phase-plot.png", width = 10, height = 6)

# get coexitance parameters

experiment_1_coexitance = experiment_1%>%
    filter(step == 1999)%>%
    filter(outcome == "Coexitance")%>%
    group_by(treatment, s_breed, f_breed)%>%
    summarise(n = n(),
              p = n/10)%>%
    ungroup()%>%
    # get mean coexitance parameters
    group_by(treatment)%>%
    summarise(mean_p = mean(p),
              sd_p = sd(p),
                max_p = max(p),
                min_p = min(p),
              mean_s_breed = mean(s_breed),
                max_s_breed = max(s_breed),
                min_s_breed = min(s_breed),
                mean_f_breed = mean(f_breed),
                max_f_breed = max(f_breed),
                min_f_breed = min(f_breed))%>%
    ungroup()%>%
    complete(treatment, fill = list(mean_p = 0, sd_p = 0, max_p = 0, min_p = 0, mean_s_breed = 0, max_s_breed = 0, min_s_breed = 0, mean_f_breed = 0, max_f_breed = 0, min_f_breed = 0))%>%
    pivot_longer(cols = c(mean_p, sd_p, max_p, min_p, mean_s_breed, max_s_breed, min_s_breed, mean_f_breed, max_f_breed, min_f_breed),
                 names_to = "parameter",
                 values_to = "value")%>%
                 pivot_wider(names_from = "treatment",
                             values_from = "value")

print("Coexitance parameters")
print(experiment_1_coexitance)

# plot outcome by treatment

experiment_1_outcome = experiment_1%>%
    group_by(treatment, s_breed, f_breed, outcome)%>%
    summarise(n = n(),
              p = n/10)%>%
    ungroup()%>%
    select(treatment, s_breed, f_breed, outcome, p)%>%
    complete(treatment, s_breed, f_breed, outcome, fill = list(p = 0))%>%
    # set order of outcome
    mutate(outcome = factor(outcome, levels = c("Prey Only", "Coexitance", "Extinction")))

ggplot(experiment_1_outcome, aes(x = s_breed, y = f_breed, fill = p))+
    geom_tile()+
    theme_bw()+
    theme(text = element_text(size = 25),
    legend.position = "top")+
    # make scale 0-1
    scale_fill_viridis(limits = c(0,1), breaks = seq(0, 1, 0.5),
    guide = guide_colorbar(barwidth = 10, barheight = 2))+
    scale_x_continuous(breaks = seq(0, 1, 0.2), expand = c(0, 0))+
    scale_y_continuous(breaks = seq(0, 1, 0.2), expand = c(0,0))+
    labs(x = "Predator breeding rate",
            y = "Prey breeding rate")+
    facet_grid(treatment~outcome)

ggsave("output/experiments/plots/experiment_1_outcome.png", width = 15, height = 12)

# Experiment 2

experiment_2$lethality = c(rep("Non-lethal", 12000), rep("Lethal", 12000))

experiment_2$target = c(rep("Prey", 4000), rep("Predator", 4000), rep("Both", 4000))

# scores

experiment_2 = experiment_2%>%
mutate(outcome = ifelse(Prey == 0, "Extinction", 
                        ifelse(Predator == 0, "Prey Only", "Coexitance")))%>%
# score
mutate(score = ifelse(outcome == "Extinction", -1, 
                      ifelse(outcome == "Prey Only", 0, 0)))

# Number of simulations that ran to completion

print("Number of simulations that ran to completion")
print(sum(experiment_2$step == 1999))

# plot N_pred vs N_prey

experiment_2%>%
filter(Predator > 0, Prey > 0, Predator + Prey < 100000)%>%
ggplot(aes(x = Predator, y = Prey, col = f_breed/s_breed))+
    geom_point(size = 3)+
    labs(x = "Predator",
            y = "Prey",
            color = "Prey:Predator breeding rate")+
    scale_color_viridis()+
    theme_bw()+
    theme(text = element_text(size = 20),
    legend.position = "top")+
    facet_grid(lethality~target)

ggsave("output/experiments/plots/experiment_2_predator-prey.png", width = 15, height = 12)

# plot mean score by treatment for s_breed vs f_breed

experiment_2_mean = experiment_2%>%
    group_by(lethality, target, s_breed, f_breed)%>%
    summarise(mean_score = mean(score))

ggplot(experiment_2_mean, aes(x = s_breed, y = f_breed, fill = mean_score))+
    geom_tile(col = "black")+
    theme_bw()+
    theme(text = element_text(size = 20),
    legend.position = "top")+
    scale_fill_viridis(limits = c(-1,1), breaks = seq(-1, 1, 0.5),
    guide = guide_colorbar(barwidth = 10, barheight = 2))+
    scale_x_continuous(breaks = seq(0, 1, 0.1))+
    scale_y_continuous(breaks = seq(0, 1, 0.1))+
    labs(x = "s_breed",
            y = "f_breed")+
    facet_grid(lethality~target)

ggsave("output/experiments/plots/experiment_2_phase-plot.png", width = 15, height = 12)

# plot outcome by treatment

experiment_2_outcome = experiment_2%>%
    group_by(lethality, target, s_breed, f_breed, outcome)%>%
    summarise(n = n(),
              p = n/10)%>%
    ungroup()%>%
    select(lethality, target, s_breed, f_breed, outcome, p)%>%
    complete(lethality, target, s_breed, f_breed, outcome, fill = list(p = 0))%>%
    # set order of outcome
    mutate(outcome = factor(outcome, levels = c("Prey Only", "Coexitance", "Extinction")))

experiment_2_lethal_outcome = experiment_2_outcome%>%
    filter(lethality == "Lethal")

ggplot(experiment_2_lethal_outcome, aes(x = s_breed, y = f_breed, fill = p))+
    geom_tile(col = "black")+
    theme_bw()+
    theme(text = element_text(size = 20),
    legend.position = "top")+
    # make scale 0-1
    scale_fill_viridis(limits = c(0,1), breaks = seq(0, 1, 0.5),
    guide = guide_colorbar(barwidth = 10, barheight = 2))+
    scale_x_continuous(breaks = seq(0, 1, 0.1), expand = c(0, 0))+
    scale_y_continuous(breaks = seq(0, 1, 0.1), expand = c(0,0))+
    labs(x = "Predator breeding rate",
            y = "Prey breeding rate")+
    facet_grid(target~outcome)

ggsave("output/experiments/plots/experiment_2_lethal_outcome.png", width = 15, height = 18)

experiment_2_non_lethal_outcome = experiment_2_outcome%>%
    filter(lethality == "Non-lethal")

ggplot(experiment_2_non_lethal_outcome, aes(x = s_breed, y = f_breed, fill = p))+
    geom_tile(col = "black")+
    theme_bw()+
    theme(text = element_text(size = 20),
    legend.position = "top")+
    # make scale 0-1
    scale_fill_viridis(limits = c(0,1), breaks = seq(0, 1, 0.5),
    guide = guide_colorbar(barwidth = 10, barheight = 2))+
    scale_x_continuous(breaks = seq(0, 1, 0.1), expand = c(0, 0))+
    scale_y_continuous(breaks = seq(0, 1, 0.1), expand = c(0,0))+
    labs(x = "Predator breeding rate",
            y = "Prey breeding rate")+
    facet_grid(target~outcome)

ggsave("output/experiments/plots/experiment_2_non_lethal_outcome.png", width = 15, height = 18)

# get coexitance parameters

experiment_2_coexitance = experiment_2%>%
    filter(step == 1999)%>%
    filter(outcome == "Coexitance")%>%
    group_by(lethality, target, s_breed, f_breed)%>%
    summarise(n = n(),
              p = n/10)%>%
    ungroup()%>%
    # get mean coexitance parameters
    group_by(lethality, target)%>%
    summarise(mean_p = mean(p),
              sd_p = sd(p),
                max_p = max(p),
                min_p = min(p),
              mean_s_breed = mean(s_breed),
                max_s_breed = max(s_breed),
                min_s_breed = min(s_breed),
                mean_f_breed = mean(f_breed),
                max_f_breed = max(f_breed),
                min_f_breed = min(f_breed))%>%
    ungroup()%>%
    complete(lethality, target, fill = list(mean_p = 0, sd_p = 0, max_p = 0, min_p = 0, mean_s_breed = 0, max_s_breed = 0, min_s_breed = 0, mean_f_breed = 0, max_f_breed = 0, min_f_breed = 0))%>%
    pivot_longer(cols = c(mean_p, sd_p, max_p, min_p, mean_s_breed, max_s_breed, min_s_breed, mean_f_breed, max_f_breed, min_f_breed),
                 names_to = "parameter",
                 values_to = "value")%>%
                 pivot_wider(names_from = "target",
                             values_from = "value")

print("Coexitance parameters")

print(experiment_2_coexitance)

# parameter for coexitance

experiment_2%>%
    filter(step == 1999)%>%
    filter(outcome == "Coexitance")%>%
    group_by(lethality, target, s_breed, f_breed)%>%
    summarise(n = n(),
              p = n/10)%>%
    ungroup()%>%
    filter(p == max(p))
