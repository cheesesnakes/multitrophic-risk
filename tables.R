pacman::p_load("dplyr", "tidyr", "stringr", "flextable", "here")

here::i_am("tables.R")
Scenarios <- 0:7

# Create table for parameter summary

for (i in Scenarios) {
    assign(
        paste0("Scenario_", i),
        read.csv(paste0("output/experiments/outcomes/Scenario-", i, "_param_summary.csv"), stringsAsFactors = FALSE)
    )
}

# Assign a scenario column to each scenario

for (i in Scenarios) {
    df <- get(paste0("Scenario_", i))
    df$Scenario <- i
    assign(paste0("Scenario_", i), df)
}

# Combine all scenarios into one data frame
all_scenarios <- do.call(rbind, lapply(Scenarios, function(i) get(paste0("Scenario_", i))))

for (i in Scenarios) {
    rm(list = paste0("Scenario_", i))
}

# Rename variables

all_scenarios <- all_scenarios %>%
    mutate(
        variable = case_when(
            variable == "f_breed" ~ "Prey Birth Rate",
            variable == "s_breed" ~ "Predator Death Rate"
        ),
        Scenario = case_when(
            Scenario == 0 ~ "Mesopredators target prey",
            Scenario == 1 ~ "Apex predators target mesopredators",
            Scenario == 2 ~ "Prey respond to non-lethal superpredators",
            Scenario == 3 ~ "Mesopredators respond to non-lethal superpredators",
            Scenario == 4 ~ "Prey and Mesopredators respond to non-lethal superpredators",
            Scenario == 5 ~ "Superperedators target prey",
            Scenario == 6 ~ "Superpredators target mesopredators",
            Scenario == 7 ~ "Superpredators target prey and mesopredators"
        )
    )

# Create a flextable

tableb2 <- all_scenarios %>%
    rename(state = phase) %>%
    select(Scenario, variable, state, mean, lower, upper) %>%
    arrange(Scenario, variable, state) %>%
    flextable() %>%
    set_header_labels(
        Scenario = "Scenario",
        variable = "Variable",
        state = "State",
        mean = "Mean",
        lower = "5th Percentile",
        upper = "95th Percentile"
    ) %>%
    # round to 3 decimal places
    set_formatter(
        mean = function(x) format(round(x, 3), nsmall = 3),
        lower = function(x) format(round(x, 3), nsmall = 3),
        upper = function(x) format(round(x, 3), nsmall = 3)
    ) %>%
    set_table_properties(
        width = 0.8,
        layout = "fixed"
    ) %>%
    # merge cells
    merge_v(~ Scenario + variable) %>%
    theme_box()

# Save the flextable as a Word document
tableb2 %>%
    save_as_docx(path = here("output", "table_B2.docx"))

# Make table for phase summary

for (i in Scenarios) {
    assign(
        paste0("Scenario_", i),
        read.csv(paste0("output/experiments/outcomes/Scenario-", i, "_phase_summary.csv"), stringsAsFactors = FALSE)
    )
}

# Assign a scenario column to each scenario
for (i in Scenarios) {
    df <- get(paste0("Scenario_", i))
    df$Scenario <- i
    assign(paste0("Scenario_", i), df)
}

# Combine all scenarios into one data frame
all_scenarios <- do.call(rbind, lapply(Scenarios, function(i) get(paste0("Scenario_", i))))

for (i in Scenarios) {
    rm(list = paste0("Scenario_", i))
}

# Rename variables

all_scenarios <- all_scenarios %>%
    mutate(
        Scenario = case_when(
            Scenario == 0 ~ "Mesopredators target prey",
            Scenario == 1 ~ "Apex predators target mesopredators",
            Scenario == 2 ~ "Prey respond to non-lethal superpredators",
            Scenario == 3 ~ "Mesopredators respond to non-lethal superpredators",
            Scenario == 4 ~ "Prey and Mesopredators respond to non-lethal superpredators",
            Scenario == 5 ~ "Superperedators target prey",
            Scenario == 6 ~ "Superpredators target mesopredators",
            Scenario == 7 ~ "Superpredators target prey and mesopredators"
        )
    )


# Create a flextable

tableb1 <- all_scenarios %>%
    rename(state = phase) %>%
    select(Scenario, state, mean, lower, upper) %>%
    arrange(Scenario, state) %>%
    flextable() %>%
    set_header_labels(
        Scenario = "Scenario",
        state = "State",
        mean = "Mean",
        lower = "5th Percentile",
        upper = "95th Percentile"
    ) %>%
    # round to 3 decimal places
    set_formatter(
        mean = function(x) format(round(x, 3), nsmall = 3),
        lower = function(x) format(round(x, 3), nsmall = 3),
        upper = function(x) format(round(x, 3), nsmall = 3)
    ) %>%
    set_table_properties(
        width = 0.8,
        layout = "fixed"
    ) %>%
    # merge cells
    merge_v(~Scenario) %>%
    theme_box()

save_as_docx(
    tableb1,
    path = here("output", "table_B1.docx")
)

# make table for periodicity summary

for (i in Scenarios) {
    assign(
        paste0("Scenario_", i),
        read.csv(paste0("output/experiments/outcomes/Scenario-", i, "_periodicity_summary.csv"), stringsAsFactors = FALSE)
    )
}

# Assign a scenario column to each scenario

for (i in Scenarios) {
    df <- get(paste0("Scenario_", i))
    df$Scenario <- i
    assign(paste0("Scenario_", i), df)
}

# Combine all scenarios into one data frame

all_scenarios <- do.call(rbind, lapply(Scenarios, function(i) get(paste0("Scenario_", i))))

for (i in Scenarios) {
    rm(list = paste0("Scenario_", i))
}



# Rename variables

all_scenarios <- all_scenarios %>%
    mutate(
        Scenario = case_when(
            Scenario == 0 ~ "Mesopredators target prey",
            Scenario == 1 ~ "Apex predators target mesopredators",
            Scenario == 2 ~ "Prey respond to non-lethal superpredators",
            Scenario == 3 ~ "Mesopredators respond to non-lethal superpredators",
            Scenario == 4 ~ "Prey and Mesopredators respond to non-lethal superpredators",
            Scenario == 5 ~ "Superperedators target prey",
            Scenario == 6 ~ "Superpredators target mesopredators",
            Scenario == 7 ~ "Superpredators target prey and mesopredators"
        )
    )

head(all_scenarios)

# Create a flextable

tableb3 <- all_scenarios %>%
    select(Scenario, mean, q_05, q_25, q_75, q_95) %>%
    arrange(Scenario) %>%
    flextable() %>%
    set_header_labels(
        Scenario = "Scenario",
        mean = "Mean",
        q_05 = "5th Percentile",
        q_25 = "25th Percentile",
        q_75 = "75th Percentile",
        q_95 = "95th Percentile"
    ) %>%
    # round to 3 decimal places
    set_formatter(
        mean = function(x) format(round(x, 3), nsmall = 3),
        q_05 = function(x) format(round(x, 3), nsmall = 3),
        q_25 = function(x) format(round(x, 3), nsmall = 3),
        q_75 = function(x) format(round(x, 3), nsmall = 3),
        q_95 = function(x) format(round(x, 3), nsmall = 3)
    ) %>%
    set_table_properties(
        width = 0.8,
        layout = "fixed"
    ) %>%
    # merge cells
    merge_v(~Scenario) %>%
    theme_box()

# Save the flextable as a Word document

tableb3 %>%
    save_as_docx(path = here("output", "table_B3.docx"))
