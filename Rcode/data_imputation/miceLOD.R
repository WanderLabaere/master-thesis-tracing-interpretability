# ---------------------------------------------------------
# miceLOD.R
# ---------------------------------------------------------
# Execution script for handling left-censored missing data (values below the 
# Limit of Detection, LOD) in geochemical datasets. The pipeline utilizes 
# Multiple Imputation by Chained Equations (MICE) paired with truncated normal 
# sampling to strictly bound imputed values between zero and the feature-specific LOD.
#
# Input data:
# 1. Filtered multi-element CSV datasets (Soil and Cocoa ICPMS) where 
#    left-censored values are explicitly denoted by a "<" prefix.
#
# Generates and saves:
# 1. Fully imputed and pooled datasets (CSV) where all missing and LOD-censored 
#    observations have been successfully replaced with valid numeric estimates.
# ---------------------------------------------------------


# --- Load libraries ---
library(readr)
library(dplyr)
library(mice)
library(truncnorm)

# --- Helper functions ---
detect_lod <- function(df_num) {
  lod <- numeric(ncol(df_num))
  names(lod) <- names(df_num)
  for(j in seq_along(df_num)){
    col <- df_num[[j]]
    censored <- grepl("^<", col)
    if(any(censored)){
      lod[j] <- as.numeric(sub("<", "", col[censored][1]))
      col[censored] <- NA
    } else {
      lod[j] <- NA
    }
    df_num[[j]] <- as.numeric(col)
  }
  list(df_num = df_num, lod = lod)
}

apply_truncation <- function(completed, df_num, lod) {
  for(v in names(lod)[!is.na(lod)]){
    idx <- which(is.na(df_num[[v]]))
    if(length(idx) > 0){
      completed[[v]][idx] <- rtruncnorm(
        n    = length(idx),
        a    = 0,
        b    = lod[v],
        mean = completed[[v]][idx],
        sd   = sd(completed[[v]], na.rm = TRUE)
      )
    }
  }
  completed
}

impute_and_pool <- function(df_num, lod, m = 5) {
  # Run MICE
  imp <- mice(df_num, method = "pmm", m = m, maxit = 10, seed = 123)
  
  # Extract all m completed datasets, apply truncation to each
  completed_list <- lapply(1:m, function(i) {
    completed <- complete(imp, i)
    apply_truncation(completed, df_num, lod)
  })
  
  # Pool by averaging across all m datasets
  # This is valid for point estimates of continuous variables
  pooled <- completed_list[[1]]
  for(i in 2:m){
    pooled <- pooled + completed_list[[i]]
  }
  pooled / m
}

# --- cI ---
cI <- read_csv("C:/Users/wande/Documents/Bioinformatics_2mas_2025-2026/Master_Thesis/Pcode/data_preparation/output/dataframes/filtered/NA_0_pct/cI_filtered_NA_0_pct.csv")
cI_num <- cI[, 17:ncol(cI)]
lod_result_cI <- detect_lod(cI_num)
cI_num <- lod_result_cI$df_num
lod_cI <- lod_result_cI$lod

completed_cI <- impute_and_pool(cI_num, lod_cI)
completed_cI_full <- cbind(cI[, 1:16], completed_cI)

# --- sI ---
sI <- read_csv("C:/Users/wande/Documents/Bioinformatics_2mas_2025-2026/Master_Thesis/Pcode/data_preparation/output/dataframes/filtered/NA_0_pct/sI_filtered_NA_0_pct.csv")
sI_num <- sI[, 17:ncol(sI)]
lod_result_sI <- detect_lod(sI_num)
sI_num <- lod_result_sI$df_num
lod_sI <- lod_result_sI$lod

completed_sI <- impute_and_pool(sI_num, lod_sI)
completed_sI_full <- cbind(sI[, 1:16], completed_sI)

# --- Save ---
cI_path <- "C:/Users/wande/Documents/Bioinformatics_2mas_2025-2026/Master_Thesis/Pcode/data_preparation/output/dataframes/imputed/miceLOD/cI_imputed_mice.csv"
sI_path <- "C:/Users/wande/Documents/Bioinformatics_2mas_2025-2026/Master_Thesis/Pcode/data_preparation/output/dataframes/imputed/miceLOD/sI_imputed_mice.csv"
write_csv(completed_cI_full, cI_path)
write_csv(completed_sI_full, sI_path)



