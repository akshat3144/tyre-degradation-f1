Sys.setenv(RTOOLS45_HOME = "C:/rtools45")
Sys.setenv(PATH = paste("C:/rtools45/x86_64-w64-mingw32.static.posix/bin",
                        "C:/rtools45/usr/bin", Sys.getenv("PATH"), sep = ";"))
.libPaths(c(file.path(Sys.getenv("LOCALAPPDATA"),"R","win-library","4.6"), .libPaths()))
suppressPackageStartupMessages({library(tidyverse); library(cmdstanr)})
set_cmdstan_path(file.path(Sys.getenv("USERPROFILE"), ".cmdstan", "cmdstan-2.39.0"))

# ---- their data prep, faithfully ----
allLaps <- read_csv("Hamilton_2025_All_Races.csv", show_col_types = FALSE) %>%
  mutate(LapTime = LapTimeSeconds) %>%
  filter(!str_detect(TrackStatus, "4|5|6|7"))

df <- allLaps %>%
  filter(Driver == "HAM", Race == "Austrian Grand Prix",
         is.na(PitOutTime) & is.na(PitInTime)) %>%
  mutate(Compound_Code = as.integer(factor(Compound, levels = c("HARD","MEDIUM","SOFT"))))

n <- nrow(df)
pit <- rep(0, n)
for (i in 2:n) if (df$Stint[i] != df$Stint[i-1]) pit[i-1] <- 1

lap_max <- max(df$LapNumber)
fuel_theirs <- seq(110, 1, length.out = n)                                   # their code
fuel_fixed  <- 110 - (110 - 1) * (df$LapNumber - 1) / (lap_max - 1)          # indexed to real laps

cat(sprintf("laps retained %d | race length %d | fuel error: max %.1f kg, mean %.1f kg\n\n",
            n, lap_max, max(abs(fuel_theirs - fuel_fixed)), mean(abs(fuel_theirs - fuel_fixed))))

cmap <- sort(unique(df$Compound_Code))
base <- list(TT = n, y = df$LapTime, Pit = pit, sdo0 = .3,
             C = 3, Compound = df$Compound_Code,
             C_used = length(cmap), compound_map = cmap,
             z_reset0 = c(69.5, 69.0, 68.5))

mod <- cmdstan_model("full_race_1driver_and_fuel_Extension1.stan")

run <- function(fuel, label) {
  d <- c(base, list(fuel_mass = fuel))
  f <- mod$sample(data = d, chains = 4, iter_warmup = 2000, iter_sampling = 2000,
                  parallel_chains = 4, seed = 1, refresh = 0, show_messages = FALSE)
  s <- f$summary(c("v_used","gamma"), "mean",
                 q = ~quantile(., probs = c(.025, .975)), "rhat")
  dr <- f$draws("v_used", format = "df")
  pmed <- mean(dr$`v_used[2]` > dr$`v_used[1]`)
  hi_h <- quantile(dr$`v_used[1]`, .975); lo_m <- quantile(dr$`v_used[2]`, .025)
  s$variable <- c("v Hard","v Medium","gamma")
  s$fuel <- label
  s$p_medium_gt_hard <- pmed
  s$cris_overlap <- lo_m < hi_h
  cat("=====", label, "=====\n")
  print(as.data.frame(s) %>% mutate(across(where(is.numeric), ~round(., 4))))
  cat(sprintf("  P(v_Medium > v_Hard) = %.3f   | 95%% CrIs overlap: %s\n\n",
              pmed, if (lo_m < hi_h) "YES" else "NO"))
  s
}

# NOTE: the canonical, tested reproduction is now Python
# (scripts/paper/p4_benchmark.py -> paper/results/benchmark.json). This script
# used to end with `out <- NULL; write_csv(out, ...)`, so it never wrote a result.
out <- bind_rows(run(fuel_theirs, "their_ramp"), run(fuel_fixed, "corrected_fuel"))
write_csv(out, "results/table3_reproduction_R.csv")
cat("WROTE results/table3_reproduction_R.csv\n")
cat("PAPER Table 3: Hard 0.054 [0.004, 0.133] | Medium 0.060 [0.009, 0.120]\n")
