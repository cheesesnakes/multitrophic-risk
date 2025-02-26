# load libraries

library(pacman)
p_load(dplyr, tidyr, ggplot2, viridis, cowplot, gridExtra, lme4, RColorBrewer)


# get sim data

data <- read.csv("output/experiments/results/Experiment-6_results.csv")

colnames(data)
head(data)

data <- data %>%
  select(c(rep_id, sample_id, f_max, step, Prey))

# add lattice size

n <- 2000 * 10 * 8

data$L <- c(rep(c(10, 20, 50), each = n), rep(100, 2000 * 10 * 7))

# create time series plot

rep_5 <- data %>%
  filter(rep_id == 5) %>%
  filter(L == 50) %>%
  mutate(f_max = as.factor(f_max))

ggplot(data = rep_5, aes(x = step, y = Prey, color = f_max)) +
  geom_line() +
  scale_color_viridis(discrete = TRUE) +
  theme_bw() +
  labs(
    x = "Time",
    y = "Number of prey"
  ) +
  theme(
    legend.position = "top",
    text = element_text(size = 20)
  ) +
  facet_wrap(~f_max, scales = "free_y")

ggsave(last_plot(), filename = "output/experiments/plots/prey-time-series.png", width = 10, height = 10, dpi = 300)

# plot single sample at L = 50, f_max = 5

fixed <- data %>%
  filter(L == 50) %>%
  filter(f_max == 5)

fixed_rep_5 <- fixed %>%
  filter(rep_id == 5)

ggplot(data = fixed, aes(x = step, y = Prey)) +
  geom_line(data = fixed_rep_5, aes(x = step, y = Prey), color = "red") +
  geom_line(alpha = 0.25) +
  theme_bw() +
  labs(
    x = "Time",
    y = "Number of prey"
  ) +
  theme(
    legend.position = "top",
    text = element_text(size = 20)
  )

ggsave(last_plot(), filename = "output/experiments/plots/prey-time-series-single.png", width = 10, height = 10, dpi = 300)

# remove the first 500

data <- data %>%
  filter(step > 500)

# calculate k^

K_est <- data %>%
  group_by(f_max, L) %>%
  summarise(
    k_est = mean(Prey),
    sd = sd(Prey)
  )

print(K_est %>%
  select(f_max, L, k_est) %>%
  pivot_wider(names_from = f_max, values_from = k_est))

# plot number of prey by f_max

ggplot(data, aes(x = f_max, y = Prey, color = as.factor(L))) +
  geom_jitter(width = 0.1, alpha = 0.5) +
  #    geom_pointrange(data = K_est, aes(y = k_est, ymin = k_est - sd, ymax = k_est + sd), size = 1) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 2, linetype = "dashed") +
  scale_color_viridis(discrete = TRUE, name = "Lattice Size") +
  theme_bw() +
  labs(
    x = "Local Saturation",
    y = "Number of prey"
  ) +
  theme(
    legend.position = "top",
    text = element_text(size = 20)
  )

ggsave(last_plot(), filename = "output/experiments/plots/carrying-capacity.png", width = 10, height = 10, dpi = 300)
