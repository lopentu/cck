library(magrittr)
library(dplyr)
library(tidyr)
library(mgcv)
library(ggplot2)

set.seed(9)

df <- read.csv("20211110-111217/gam_df2.csv") %>%
    separate(dzg, into=c("dzg_lev1", "dzg_lev2"), sep="/", remove=FALSE) %>%
    mutate(char=char_id)

colnames(df)

# set up gam
ctrl <- gam.control(trace = TRUE)
df$char <- factor(df$char)
df$author_norm <- factor(df$author_norm)
df$dzg <- factor(df$dzg)
df$dzg_lev1 <- factor(df$dzg_lev1)

# # gam model 1 from last meeting
# gam_mod1 <- bam(logfreq ~ s(k, char, bs="fs", m=1), data=df, control=ctrl, discrete=TRUE)
# saveRDS(gam_mod1, "dat_gam.rds")

# # gam model 2 from last meeting
# gam_mod2 <- bam(logfreq ~ char + s(k, by=char, id=1, m=1), data=df, control=ctrl, discrete=TRUE)
# saveRDS(gam_mod2, "dat_gam2.rds")

# # gam_mod3 <- bam(raw_freq ~ s(k, char, bs="fs", m=1), family="poisson",
# #                 data=df, control=ctrl, discrete=TRUE)
# # # discretization only available with fREML
# # saveRDS(gam_mod3, "dat_gam3.rds")

# # gam_mod4 <- bam(raw_freq ~ char + s(k, by=char, id=1, m=1), family="poisson",
# #                 data=df, control=ctrl, discrete=TRUE)
# # saveRDS(gam_mod4, "dat_gam4.rds")

# # gam_mod5 <- bam(raw_freq ~ char + s(k, by=char, id=1, m=1) + s(mid_year, by=char, id=1, m=1), family="poisson",
# #                 data=df, control=ctrl, discrete=TRUE)
# # saveRDS(gam_mod5, "dat_gam5.rds")

# # gam_mod5b <- bam(raw_freq ~ char + s(k, by=char, id=1, m=1) + s(mid_year_scaled, by=char, id=1, m=1), family="poisson",
# #                 data=df, control=ctrl, discrete=TRUE)
# # saveRDS(gam_mod5b, "dat_gam5b.rds")

# gam_mod6 <- bam(raw_freq ~ char + s(k, by=char, id=1, m=1) + s(mid_year, by=char, id=1, m=1), family=ziP(),
#                 data=df, control=ctrl, discrete=TRUE)
# saveRDS(gam_mod6, "dat_gam6.rds")

# # gam_mod7 <- bam(logfreq ~ char + s(k, by=char, id=1, m=1) + author_norm, data=df, control=ctrl, discrete=TRUE)
# # saveRDS(gam_mod7, "dat_gam7.rds")

# # gam_mod8 <- bam(logfreq ~ char + s(k, by=char, id=1, m=1) + s(mid_year_scaled, by=char, id=1, m=1) + author_norm, data=df, control=ctrl, discrete=TRUE)
# # saveRDS(gam_mod8, "dat_gam8.rds")

# # gam_mod9 <- bam(logfreq ~ char + s(k, by=char, id=1, m=1) + s(mid_year_scaled, by=char, id=1, m=1) + author_norm + dzg, data=df, control=ctrl, discrete=TRUE)
# # saveRDS(gam_mod9, "dat_gam9.rds")

# gam_mod10 <- bam(raw_freq ~ s(k, char, bs="fs", m=1),
#                  data=df, control=ctrl, discrete=TRUE, family=ziP())
# saveRDS(gam_mod10, "dat_gam10.rds")

# gam_mod11 <- bam(raw_freq ~ char + s(k, by=char, id=1, m=1), family=ziP(),
#                  data=df, control=ctrl, discrete=TRUE)
# saveRDS(gam_mod11, "dat_gam11.rds")

# gam_mod12 <- bam(raw_freq ~ char + s(k, by=char, id=1, m=1) + author_norm, data=df, control=ctrl, discrete=TRUE)
# saveRDS(gam_mod12, "dat_gam12.rds")

# gam_mod13 <- bam(raw_freq ~ char + s(k, by=char, id=1, m=1) + author_norm + dzg, data=df, control=ctrl, discrete=TRUE)
# saveRDS(gam_mod13, "dat_gam13.rds")

# gam_mod13 <- bam(raw_freq ~ char + s(k, by=char, id=1, m=1) + author_norm + dzg, data=df, control=ctrl, discrete=TRUE)
# saveRDS(gam_mod13, "dat_gam13.rds")

# gam_mod14 <- bam(raw_freq ~ char + s(k, by=char, id=1, m=1) + author_norm + dzg_lev1, data=df, control=ctrl, discrete=TRUE)
# saveRDS(gam_mod14, "dat_gam14.rds")

# from Yu-Ying
gam_mod15 <- bam(raw_freq ~ s(k, by=char, id=1, m=1) + s(char, bs="re"), family=ziP(), data=df, control=ctrl, discrete=TRUE)
saveRDS(gam_mod15, "dat_gam15.rds")

gam_mod16 <- bam(raw_freq ~ s(k, char, bs="fs", m=1, k=20), family="poisson", data=df, control=ctrl, discrete=TRUE)
saveRDS(gam_mod16, "dat_gam16.rds")

gam_mod17 <- bam(raw_freq ~ s(k, char, bs="fs", m=1, k=20) + s(author_norm, bs="re"), family="poisson", data=df, control=ctrl, discrete=TRUE)
saveRDS(gam_mod17, "dat_gam17.rds")

gam_mod18 <- gamm(logfreq ~ s(k, char, m=1, bs="fs", k=20), data=df)
saveRDS(gam_mod18, "dat_gam18.rds")

models <- lapply(list.files(".", "*.rds"), readRDS)
lapply(models, function(x){x$formula})
sapply(models, AIC)

# summary(gam_mod1)
# coef(gam_mod1)
# plot(gam_mod1, all.terms=TRUE)

# k and mid_year have partial effects in opposite directions. Yet, mid_year has a stronger effect than k