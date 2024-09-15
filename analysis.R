
# load libraries

library(pacman)
p_load(dplyr, tidyr, ggplot2, viridis, cowplot, gridExtra)


# loop through models, skip if file does not exist

super = read.csv("output/super_results.csv")
apex = read.csv("output/apex_results.csv")
lv = read.csv("output/lv_results.csv")

super$model = "super"
apex$model = "apex"
lv$model = "lv"

max_steps = 999

# add missing columns

apex$super_target = NA
apex$super_lethality = NA

lv$super_target = NA
lv$super_lethality = NA

# combine data

data = rbind(super, apex, lv)

# summarise data

head(data)
summary(data)

# add scores

data_scored = data%>%
mutate(outcome = ifelse(Predator == 0, "Prey Wins", 
                            ifelse(Prey == 0, "Extinction", 
                                    "Coexistance")),
        outcome = ifelse(step == max_steps, outcome, ifelse(outcome == "Coexistance" & Predator > Prey, "Extinction", outcome)),
        score = ifelse(outcome == "Prey Wins", 0, 
                        ifelse(outcome == "Extinction", -1, 1)))


# relationship between model parameters and outcome

## prameaters: 
## lv: s_breed, f_breed, f_die, s_energy
## apex: s_breed, f_breed, f_die, s_energy
## super: s_breed, f_breed, f_die, s_energy, super_target, super_lethality

# plot 1: mean score by s_breed vs f_breed for each f_die, grid for each s_energy

# lv model

data_scored%>%
    filter(model == "lv")%>%
    group_by(s_breed, f_breed, f_die, s_energy)%>%
    summarise(mean_score = mean(score), sd_score = sd(score), n = n())%>%
    ggplot(aes(x = s_breed, y = f_breed, fill = mean_score))+
    geom_tile()+
    facet_grid(f_die~s_energy)+
    theme_bw()+
    scale_fill_viridis(name = "Outcome", 
    # limits for color scale
    limits = c(-1, 1)
    )+
    theme(legend.position = "top",
    text = element_text(size =  20),
    # legend width
    legend.key.width = unit(2, "cm"),
    # legend height
    legend.key.height = unit(0.5, "cm"))+
    labs(x = "Predator Breeding Rate", y = "Prey Breeding Rate")
  
ggsave("output/figures/lv_params.png", width = 10, height = 10, dpi = 300)

# apex model

data_scored%>%
    filter(model == "apex")%>%
    group_by(s_breed, f_breed, f_die, s_energy)%>%
    summarise(mean_score = mean(score), sd_score = sd(score), n = n())%>%
    ggplot(aes(x = s_breed, y = f_breed, fill = mean_score))+
    geom_tile()+
    facet_grid(f_die~s_energy)+
    theme_bw()+
    scale_fill_viridis(name = "Outcome", limits = c(-1, 1))+
    theme(legend.position = "top",
    text = element_text(size =  20),
    # legend width
    legend.key.width = unit(2, "cm"),
    # legend height
    legend.key.height = unit(0.5, "cm"))+
    labs(x = "Predator Breeding Rate", y = "Prey Breeding Rate")

ggsave("output/figures/apex_params.png", width = 10, height = 10, dpi = 300)

# super model

# create grid for super_target and super_lethality

super_target = c("Prey", "Predator", "Both")
super_lethality = c("Lethal", "Non-lethal")

super_data_scored = data_scored%>%
    filter(model == "super")%>%
    mutate(super_target = ifelse(super_target == 1, "Prey", 
                                 ifelse(super_target == 2, "Predator", "Both")),
        super_lethality = ifelse(super_lethality == 1, "Lethal", "Non-lethal"))%>%
    group_by(s_breed, f_breed, f_die, s_energy, super_target, super_lethality)%>%
    summarise(mean_score = mean(score), sd_score = sd(score), n = n())

# create ggplot grid for each super_target and super_lethality

plots = list()

for (i in super_target){
    
    for (j in super_lethality){
        
        plot = super_data_scored%>%
            filter(super_target == i, super_lethality == j)%>%
            ggplot(aes(x = s_breed, y = f_breed, fill = mean_score))+
            geom_tile()+
            facet_grid(f_die~s_energy)+
            theme_bw()+
            scale_fill_viridis(name = "Outcome", limits = c(-1, 1))+
            theme(legend.position = "top",
            text = element_text(size =  20),
            # legend width
            legend.key.width = unit(2, "cm"),
            # legend height
            legend.key.height = unit(0.5, "cm"))+
            labs(x = "Predator Breeding Rate", y = "Prey Breeding Rate",
                title = paste("Super Model: Target = ", i, ", Lethality = ", j))+
            # center title
            theme(plot.title = element_text(hjust = 0.5))
        
        plots[[paste(i, j)]] = plot
    
    }

}

super_plot = grid.arrange(grobs = plots, ncol = 2, nrow = 3)

ggsave(super_plot, file = "output/figures/super_params.png", width = 20, height = 30, dpi = 300)
