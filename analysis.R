# load libraries

library(pacman)
p_load(dplyr, tidyr, ggplot2)

# import data

data <- read.csv("output/results.csv")

# summarise data

head(data)
summary(data)

# add scores

data_scored = data%>%
mutate(outcome = ifelse(Predator == 0, "Prey Wins", 
                            ifelse(Prey == 0, "Exctinction", 
                                    "Coexistance")),
        score = ifelse(outcome == "Prey Wins", 0, 
                        ifelse(outcome == "Extinction", -1, 1)),
        strat = ifelse(prey_info == 0 & predator_info == 0, "Naive",
                       ifelse(prey_info == 1 & predator_info == 0, "Prey Informed",
                              ifelse(prey_info == 0 & predator_info == 1, "Predator Informed",
                                     "Both Informed"))))

# mean scores

data_summary = data_scored%>%
    group_by(strat, s_energy, s_breed, f_breed, f_die)%>%
    summarise(mean_score = mean(score),
              sd_score = sd(score),
              n = n())

# plot grid of s_breed vs f_breed

ggplot(data_summary, aes(x = s_breed, y = f_breed, fill = mean_score))+
    geom_tile()+
    scale_fill_viridis_c()+
    facet_grid(strat~f_die, scales = "free")+
    theme_bw()+
    theme(text = element_text(size = 20),
    legend.position = "bottom")

ggsave("score_grid_f-die.png", height = 24, width = 12, dpi = 300)

# plot grid of s_breed vs f_breed

ggplot(data_summary, aes(x = s_breed, y = f_breed, fill = mean_score))+
    geom_tile()+
    scale_fill_viridis_c()+
    facet_grid(strat~s_energy, scales = "free")+
    theme_bw()+
    theme(text = element_text(size = 20),
    legend.position = "bottom")

ggsave("score_grid_s-energy.png", height = 24, width = 6, dpi = 300)