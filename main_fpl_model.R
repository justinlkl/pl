#!/usr/bin/env Rscript

suppressMessages({
  library(tidyverse)
  library(data.table)
  library(lubridate)
})

# Configuration - FIX: Use ifelse() instead of Sys.getenv() default
data_dir <- ifelse(
  nzchar(Sys.getenv("DATA_DIR")), 
  Sys.getenv("DATA_DIR"), 
  "../fpl_data/data"
)
output_dir <- "results"

# Create output directory
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Logging function
log_msg <- function(msg, level = "INFO") {
  utc_time <- format(Sys.time(), "%Y-%m-%d %H:%M:%S", tz = "UTC")
  hk_time <- format(Sys.time(), "%H:%M:%S", tz = "Asia/Hong_Kong")
  cat(sprintf("[%s UTC | %s HK] [%s] %s\n", utc_time, hk_time, level, msg))
}

# Main execution
log_msg("═══════════════════════════════════════════════════════")
log_msg("FPL DATA LOADING STARTED")
log_msg("═══════════════════════════════════════════════════════")

# Load data
tryCatch({
  log_msg(sprintf("Loading data from: %s", data_dir))
  
  # Find CSV files
  csv_files <- list.files(data_dir, pattern = "\\.csv$", full.names = TRUE, recursive = TRUE)
  
  if (length(csv_files) == 0) {
    log_msg(sprintf("⚠️  No CSV files found in %s", data_dir), "WARNING")
  } else {
    log_msg(sprintf("✓ Found %d CSV file(s)", length(csv_files)))
  }
  
  # Load all files
  data_list <- map(csv_files, ~{
    tryCatch({
      df <- fread(.x)
      log_msg(sprintf("✓ Loaded %s (%d rows)", basename(.x), nrow(df)))
      df
    }, error = function(e) {
      log_msg(sprintf("✗ Error loading %s: %s", basename(.x), e$message), "ERROR")
      NULL
    })
  })
  
  # Filter nulls
  data_list <- data_list[!sapply(data_list, is.null)]
  
  log_msg("✓ Data loading completed successfully")
  
  # Create summary
  summary_stats <- tibble(
    execution_time = format(Sys.time(), "%Y-%m-%d %H:%M:%S UTC", tz = "UTC"),
    hk_time = format(Sys.time(), "%Y-%m-%d %H:%M:%S", tz = "Asia/Hong_Kong"),
    data_dir_used = data_dir,
    files_loaded = length(data_list),
    total_rows = sum(sapply(data_list, nrow)),
    status = "success"
  )
  
  print(summary_stats)
  
  # Save results
  output_file <- file.path(output_dir, sprintf("data_load_%s.csv", format(Sys.time(), "%Y%m%d_%H%M%S")))
  fwrite(summary_stats, output_file)
  log_msg(sprintf("✓ Data loading summary saved: %s", output_file))
  
}, error = function(e) {
  log_msg(sprintf("✗ Error: %s", e$message), "ERROR")
  quit(status = 1)
})

log_msg("═══════════════════════════════════════════════════════")
log_msg("✓ FPL DATA LOADING COMPLETED")
log_msg("═══════════════════════════════════════════════════════")

quit(status = 0)
