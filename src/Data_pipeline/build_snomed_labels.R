
library(tidyverse)
library(jsonlite)
library(arrow) # Required for saving to Parquet

# 1. Load the database
ptbxl_database <- read_csv("data/Raw/PTBXL/ptbxl_database.csv") %>%
  select(ecg_id, scp_codes)

# 2. Parse the dictionary strings into a Tidy "Long" format
ptbxl_long <- ptbxl_database %>%
  mutate(
    # Fix single quotes for JSON compatibility
    scp_json = str_replace_all(scp_codes, "'", '"'),
    scp_list = map(scp_json, ~ fromJSON(.x))
  ) %>%
  unnest_longer(scp_list, values_to = "weight", indices_to = "scp_code") %>%
  filter(weight > 0) %>%
  select(ecg_id, scp_code)

# 3. Load and clean the SNOMED Mapping
# We keep IDs as characters to prevent precision loss!
ptbxl_to_snomed <- read_csv("data/Raw/PTBXL/ptbxlToSNOMED.csv") %>%
  select(Acronym, starts_with("id")) %>%
  mutate(across(starts_with("id"), as.character))

# 4. Join and Flatten the Hierarchy
# Since one SCP code can map to id1, id2, id3, and id4, we must collect them all
ptbxl_mapped <- ptbxl_long %>%
  left_join(ptbxl_to_snomed, by = c("scp_code" = "Acronym")) %>%
  # Pivot the id1, id2... columns into a single 'snomed_id' column
  pivot_longer(cols = starts_with("id"), 
               values_to = "snomed_id", 
               values_drop_na = TRUE) %>%
  select(ecg_id, snomed_id) %>%
  distinct() # Remove duplicates if multiple SCPs lead to same SNOMED ID

# 5. Create the Final "Wide" Matrix (One-Hot Encoding)
# This is what your Python model actually needs
ptbxl_final <- ptbxl_mapped %>%
  mutate(val = 1) %>%
  pivot_wider(names_from = snomed_id, 
              values_from = val, 
              values_fill = 0,
              names_prefix = "SNOMED_")

# 6. Save as Parquet for the Python Handoff
write_parquet(ptbxl_final, "data/Processed/ptbxl_with_snomed.parquet")

glimpse(ptbxl_final)