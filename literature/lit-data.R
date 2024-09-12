# analysing articles data for intitial screening

setwd("C:/Users/shony/Nextcloud/Work/PhD/Thesis/chapter-2/Analysis")

articles <- read.csv("articles.csv")

## cleaning --------

summary(articles)

head(articles)

library(dplyr)
library(tidyr)
library(stringr)

## function to split notes from rayyan

note_split <- function(notes) {
  
  test <- str_split(notes, "    | \\|")
  
  test <- unlist(test, use.names = T)
  
  test <- str_trim(test)
  
  test <- str_split(test, ":", simplify = T)
  
  test <- data.frame(test)
  
  test <- test%>%
    unite("value",X2:colnames(test)[ncol(test)], sep = ":")%>%
    rename(var = X1)%>%
    mutate(id = 1)
  
  test <- pivot_wider(test, id_cols = id,  names_from = var, values_from = value)
  
  return(test[2:ncol(test)])
  
}

## Splitting notes column

library(purrr)

articles <- articles%>%
  mutate(data = map(notes, ~note_split(.)))%>%
  select(-notes)%>%
  unnest(data)

## filtering articlested papers

articles <- articles[!is.na(articles$`RAYYAN-INCLUSION`),]

summary(articles)
head(articles)

## function to split reviewer and inclusion status

inc <- function(stat) {
  
  in_stat <- stat
  
  in_stat <- str_split(in_stat, "=>",simplify = T)
  
  in_stat <- str_remove_all(string = in_stat, pattern = "[:punct:]")
  
  in_stat <- str_trim(in_stat)
  
  reviewer = in_stat[1]
  
  status = in_stat[2]
  
  return(data.frame(reviewer, status))
  
}

## cleaning review status

articles <- articles%>%
  mutate(data = map(`RAYYAN-INCLUSION`, ~inc(.)))%>%
  select(-`RAYYAN-INCLUSION`)%>%
  unnest(data)

## selecting relavant vars

articles <- articles%>%
  select(-c(month, day, issn, volume, issue, pages, url, publisher, pubmed_id, pmc_id, `Times Cited in Web of Science Core Collection`, `Total Times Cited`, `Cited Reference Count`, Location))


## export for full text screening

full_text <- filter(articles, status == "Included" | status == "Maybe")

write.csv(full_text, "articles_full-text.csv", row.names = F, na = "")
