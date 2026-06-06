# ---------------------------------------------------------
# miceNA.R
# ---------------------------------------------------------
# Execution script for handling standard Missing At Random (MAR) values 
# in geochemical datasets using Multiple Imputation by Chained Equations (MICE).
# The script imputes values specifically within isolated taxonomic subsets 
# (genus-level) to respect specific biological elemental signatures.
#
# Input data:
# 1. Filtered multi-element CSV datasets (Timber XRF) containing true `NA` 
#    missing values, retaining all elemental features to provide context 
#    during the imputation process.
#
# Generates and saves:
# 1. Fully imputed datasets (CSV) where targeted critical features 
#    (e.g., Ba, Br) have been estimated, while overly sparse features 
#    are dropped post-imputation to ensure high data quality.
# ---------------------------------------------------------

# --- Load libraries ---
library(readr)
library(dplyr)
library(mice)

# --- Helper function ---
impute_and_average <- function(df_num, m = 5) {
  # 1. Run MICE to impute standard NA values
  imp <- mice(df_num, method = "pmm", m = m, maxit = 10, seed = 123, printFlag = FALSE)
  
  # 2. Extract all m completed datasets
  completed_list <- lapply(1:m, function(i) complete(imp, i))
  
  # 3. Pool by averaging across all m datasets
  pooled <- completed_list[[1]]
  for(i in 2:m){
    pooled <- pooled + completed_list[[i]]
  }
  
  # Return the averaged dataframe
  pooled / m
}

# --- Define Paths ---
input_path <- "C:/Users/wande/Documents/Bioinformatics_2mas_2025-2026/Master_Thesis/Pcode/data_preparation/output/dataframes/filtered/NA_0_pct/tX_filtered_NA_0_pct_Ba_Br.csv"
output_path <- "C:/Users/wande/Documents/Bioinformatics_2mas_2025-2026/Master_Thesis/Pcode/data_preparation/output/dataframes/imputed/miceNA/tX_imputed_NA_0_Ba_Br.csv"

# --- Execute Script ---
# 1. Load data
cat("Loading data...\n")
tX_data <- read_csv(input_path)

# 2. Identify variables with NAs to drop LATER
cat("\n=== NA Column Tracking ===\n")
# Find all columns containing at least one NA
na_counts <- colSums(is.na(tX_data))
vars_with_na <- names(na_counts[na_counts > 0])

# Define the exceptions to keep permanently
keep_na_vars <- c("Ba", "Br")

# Identify which variables to drop AFTER imputation
vars_to_drop_later <- setdiff(vars_with_na, keep_na_vars)

if (length(vars_to_drop_later) > 0) {
  cat("The following variables contain NA values and will be removed AFTER imputation:\n")
  for (v in vars_to_drop_later) {
    cat(sprintf(" - %s (%d missing values)\n", v, na_counts[v]))
  }
} else {
  cat("No variables flagged for removal (only Ba, Br, or columns with 0 NAs exist).\n")
}
cat("==========================\n\n")


# 3. NA Summary for Ba and Br per Genus (Before Imputation)
cat("\n=== Ba & Br NA Summary per Genus ===\n")

cols_to_check <- intersect(keep_na_vars, names(tX_data))

if (length(cols_to_check) > 0) {
  ba_br_summary <- tX_data %>%
    group_by(Genus) %>%
    summarise(across(all_of(cols_to_check), ~ sum(is.na(.)), .names = "{.col}_NAs"), .groups = 'drop') %>%
    filter(if_any(ends_with("_NAs"), ~ . > 0))
  
  if (nrow(ba_br_summary) > 0) {
    cat("Missing values per Genus prior to imputation:\n")
    for (i in 1:nrow(ba_br_summary)) {
      cat(sprintf(" - %s: ", ba_br_summary$Genus[i]))
      counts <- sapply(cols_to_check, function(col) {
        col_name <- paste0(col, "_NAs")
        sprintf("%s = %d", col, ba_br_summary[[col_name]][i])
      })
      cat(paste(counts, collapse = ", "), "\n")
    }
  } else {
    cat("No missing values for Ba or Br found in any Genus.\n")
  }
} else {
  cat("Columns 'Ba' and 'Br' do not exist in the dataset.\n")
}
cat("====================================\n\n")


# 4. Impute by Genus (Using fi;tered dataset)
cat("Running MICE imputation per Genus (this may take a moment)...\n")

split_by_genus <- split(tX_data, tX_data$Genus)

imputed_list <- lapply(split_by_genus, function(genus_df) {
  
  # Isolate numeric columns for this specific Genus 
  genus_num <- genus_df[, 14:ncol(genus_df)]
  
  # Check if this specific Genus has NA values
  if(sum(is.na(genus_num)) > 0) {
    completed_num <- impute_and_average(genus_num, m = 5)
    cbind(genus_df[, 1:13], completed_num)
  } else {
    genus_df
  }
})

# Recombine all the separate Genus dataframes back into one dataframe
completed_tX_full <- bind_rows(imputed_list)

# 5. Drop the tracked NA columns
if (length(vars_to_drop_later) > 0) {
  cat("\nDropping non-Ba/Br variables that originally contained NAs...\n")
  completed_tX_full <- completed_tX_full %>% select(-all_of(vars_to_drop_later))
}

# 6. Save the imputed dataset
output_dir <- dirname(output_path)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

write_csv(completed_tX_full, output_path)
cat(sprintf("\nImputation complete! Final data saved to:\n%s\n", output_path))