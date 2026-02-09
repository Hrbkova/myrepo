# =============================================================================
# PREPROCESSING PIPELINE FOR CZECH POLITICAL IDENTITY DATA
# =============================================================================
#
# This script:
# 1. Loads raw survey data
# 2. Applies improved lemmatization (preserves negation: nevím → nevědět)
# 3. Removes "don't know" responses (manual exclusion lists)
# 4. Outputs clean data for all downstream analysis (keyness, STM, MDS)
#
# Author: [Your name]
# Last updated: [Date]
# =============================================================================

# =============================================================================
# SETUP
# =============================================================================

library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(udpipe)
library(tibble)

# Set your paths
DATA_PATH <- "/Users/lenkahrbkova/Downloads/Czech_Transformed.csv"
UDPIPE_MODEL_PATH <- "/Users/lenkahrbkova/czech-pdt-ud-2.5-191206.udpipe"
OUTPUT_DIR <- "/Users/lenkahrbkova/Downloads/"

# Load UDPipe model
cat("Loading UDPipe Czech model...\n")
udmodel_czech <- udpipe_load_model(UDPIPE_MODEL_PATH)

# =============================================================================
# IMPROVED LEMMATIZATION FUNCTION
# =============================================================================
# Key improvement: Preserves negation (nevím → nevědět, NOT vědět)

lemmatize_czech_improved <- function(texts, udmodel, doc_ids = NULL,
                                      pos_keep = c("NOUN", "ADJ", "VERB", "AUX"),
                                      min_nchar = 3) {

  if (is.null(doc_ids)) doc_ids <- paste0("doc_", seq_along(texts))

  # Basic text cleaning
  texts_clean <- texts %>%
    str_to_lower() %>%
    str_squish() %>%
    str_remove_all("https?://\\S+|www\\.\\S+|\\S+@\\S+") %>%  # URLs, emails
    str_remove_all("[^a-záčďéěíňóřšťúůýž\\s]") %>%            # Keep only Czech letters
    str_squish()

  cat("Annotating", length(texts), "texts...\n")

  # UDPipe annotation
  ann <- udpipe_annotate(udmodel, x = texts_clean, doc_id = doc_ids) %>%
    as.data.frame()

  # KEY IMPROVEMENT: Detect morphological negation in features
  has_feats <- !is.na(ann$feats)
  is_neg <- has_feats & grepl("\\bPolarity=Neg\\b", ann$feats)

  # Start from lemma, then fix negated VERBs
  ann$lemma_work <- ann$lemma
  is_verb_neg <- ann$upos == "VERB" & is_neg
  ann$lemma_work[is_verb_neg] <- paste0("ne", ann$lemma[is_verb_neg])

  # FIX: Include AUX for negation (nevím can be tagged as AUX in some contexts)
  is_aux_neg <- ann$upos == "AUX" & is_neg
  ann$lemma_work[is_aux_neg] <- paste0("ne", ann$lemma[is_aux_neg])

  # Filter: keep specified POS, drop very short tokens
  ann_keep <- ann %>%
    filter(upos %in% pos_keep) %>%
    mutate(final_tok = ifelse(nchar(lemma_work) >= min_nchar, lemma_work, NA_character_)) %>%
    drop_na(final_tok)

  # Rebuild text per document
  lemma_result <- ann_keep %>%
    group_by(doc_id) %>%
    summarise(lemmatized_text = paste(final_tok, collapse = " "), .groups = "drop")

  # FIX: Preserve document order using left_join from skeleton
  skeleton <- tibble(doc_id = doc_ids)
  result <- skeleton %>%
    left_join(lemma_result, by = "doc_id") %>%
    mutate(lemmatized_text = coalesce(lemmatized_text, ""))

  return(result)
}

# =============================================================================
# STEP 1: LOAD RAW DATA
# =============================================================================

cat("\n", rep("=", 60), "\n")
cat("STEP 1: Loading raw data\n")
cat(rep("=", 60), "\n")

data_raw <- read_csv(DATA_PATH, na = c("", "NA"))

cat("Raw data loaded:", nrow(data_raw), "rows\n")
cat("Columns:", paste(names(data_raw)[1:10], collapse = ", "), "...\n")

# =============================================================================
# STEP 2: INITIAL FILTERING
# =============================================================================

cat("\n", rep("=", 60), "\n")
cat("STEP 2: Initial filtering\n")
cat(rep("=", 60), "\n")

# Keep only rows with both ingroup and outgroup responses
# NOTE: respondent_id = row_number() is based on post-filter ordering.
# The manual exclude lists (Step 4) were built using this same filtering.
# If you rebuild exclude lists from different data, IDs may misalign!
# FIX: Keep raw_rowid as stable anchor for debugging/rebuilding exclude lists
data_filtered <- data_raw %>%
  mutate(raw_rowid = row_number()) %>%
  filter(!is.na(open_ingroup) & !is.na(open_outgroup) &
           nchar(trimws(open_ingroup)) > 0 &
           nchar(trimws(open_outgroup)) > 0) %>%
  mutate(
    respondent_id = row_number(),
    ingroup_doc_id = paste0("resp_", respondent_id, "_ingroup"),
    outgroup_doc_id = paste0("resp_", respondent_id, "_outgroup")
  )

cat("After initial filter:", nrow(data_filtered), "rows\n")
cat("Removed:", nrow(data_raw) - nrow(data_filtered), "rows with missing responses\n")

# =============================================================================
# STEP 3: LEMMATIZATION
# =============================================================================

cat("\n", rep("=", 60), "\n")
cat("STEP 3: Lemmatization (with negation preservation)\n")
cat(rep("=", 60), "\n")

# Lemmatize ingroup responses
cat("\nProcessing INGROUP responses...\n")
ingroup_lemmas <- lemmatize_czech_improved(
  texts = data_filtered$open_ingroup,
  udmodel = udmodel_czech,
  doc_ids = data_filtered$ingroup_doc_id
)

# Lemmatize outgroup responses
cat("\nProcessing OUTGROUP responses...\n")
outgroup_lemmas <- lemmatize_czech_improved(
  texts = data_filtered$open_outgroup,
  udmodel = udmodel_czech,
  doc_ids = data_filtered$outgroup_doc_id
)

# Merge lemmatized text back to main data
data_lemmatized <- data_filtered %>%
  left_join(ingroup_lemmas, by = c("ingroup_doc_id" = "doc_id")) %>%
  rename(ingroup_lemma = lemmatized_text) %>%
  left_join(outgroup_lemmas, by = c("outgroup_doc_id" = "doc_id")) %>%
  rename(outgroup_lemma = lemmatized_text)

cat("\nLemmatization complete!\n")
cat("Sample lemmatized ingroup:\n")
print(head(data_lemmatized$ingroup_lemma, 3))

# =============================================================================
# STEP 4: REMOVE "DON'T KNOW" RESPONSES
# =============================================================================

cat("\n", rep("=", 60), "\n")
cat("STEP 4: Removing 'don't know' responses\n")
cat(rep("=", 60), "\n")

# Manual exclusion lists (from your careful review)
# INGROUP: IDs identified as pure "don't know" responses
exclude_ingroup_ids <- c(
  8, 15, 31, 86, 100, 198, 200, 214, 226, 250, 258, 259, 266, 273, 297,
  303, 348, 375, 430, 474, 480, 501, 520, 527, 530, 542, 544, 547, 552,
  585, 598, 620, 653, 654, 681, 709, 741, 746, 754, 756, 777, 802, 806,
  816, 824, 825, 836, 872, 881, 899, 905, 915, 965, 996, 1037, 1084,
  1115, 1154, 1188, 1199, 1212, 1224, 1271, 1275, 1276, 1370, 1427
)

# OUTGROUP: IDs identified as pure "don't know" responses
exclude_outgroup_ids <- c(
  7, 8, 15, 58, 86, 91, 100, 109, 174, 200, 209, 211, 226, 231, 250,
  258, 259, 266, 280, 297, 329, 341, 342, 373, 375, 395, 398, 452,
  474, 480, 481, 501, 516, 520, 525, 527, 530, 541, 542, 550, 552,
  557, 577, 590, 598, 620, 636, 704, 709, 714, 756, 782, 806, 815,
  816, 824, 825, 836, 849, 899, 905, 916, 965, 995, 1019, 1037, 1047,
  1119, 1130, 1154, 1188, 1194, 1211, 1212, 1224, 1247, 1275, 1281,
  1342, 1355, 1370, 1426, 1431, 1432
)

cat("Ingroup 'don't know' exclusions:", length(exclude_ingroup_ids), "\n")
cat("Outgroup 'don't know' exclusions:", length(exclude_outgroup_ids), "\n")

# Exclude if EITHER ingroup OR outgroup is "don't know"
# (ensures both responses are meaningful for comparison)
exclude_either <- unique(c(exclude_ingroup_ids, exclude_outgroup_ids))
cat("Total unique IDs to exclude:", length(exclude_either), "\n")

data_clean <- data_lemmatized %>%
  filter(!respondent_id %in% exclude_either)

cat("\nAfter removing 'don't know':", nrow(data_clean), "rows\n")
cat("Removed:", nrow(data_lemmatized) - nrow(data_clean), "rows\n")

# FIX: Drop docs that became empty after lemmatization
n_before_empty <- nrow(data_clean)
data_clean <- data_clean %>%
  mutate(
    ingroup_lemma = str_squish(ingroup_lemma),
    outgroup_lemma = str_squish(outgroup_lemma)
  ) %>%
  filter(ingroup_lemma != "" & outgroup_lemma != "")

cat("After removing empty lemmas:", nrow(data_clean), "rows\n")
cat("Removed:", n_before_empty - nrow(data_clean), "rows with empty lemmatized text\n")

# =============================================================================
# STEP 5: FINAL DATA (minimal filtering - method-specific filtering at analysis)
# =============================================================================

cat("\n", rep("=", 60), "\n")
cat("STEP 5: Final data preparation\n")
cat(rep("=", 60), "\n")

# Best practice: Don't aggressively filter short responses in preprocessing.
# Each analysis method handles this appropriately:
# - STM: prepDocuments(lower.thresh = 3) removes rare words
# - Quanteda/keyness: dfm_trim(min_termfreq = 3) for rare words
# - Sentence transformers: handles short text natively, no filtering needed

data_final <- data_clean

cat("Final sample size:", nrow(data_final), "rows\n")

# =============================================================================
# STEP 6: ADD DERIVED VARIABLES
# =============================================================================

cat("\n", rep("=", 60), "\n")
cat("STEP 6: Adding derived variables\n")
cat(rep("=", 60), "\n")

# Add party names
data_final <- data_final %>%
  mutate(
    party_name = case_when(
      party_choice == 1 ~ "ANO",
      party_choice == 2 ~ "ODS",
      party_choice == 3 ~ "SPD",
      party_choice == 4 ~ "STAN",
      party_choice == 5 ~ "Piráti",
      party_choice == 6 ~ "KDU-ČSL",
      party_choice == 7 ~ "TOP 09",
      party_choice == 8 ~ "Jiná strana",
      party_choice == 9 ~ "Nevím",
      party_choice == 10 ~ "Nešel/a bych volit",
      party_choice == 11 ~ "KSČM/Stačilo",
      TRUE ~ "Unknown"
    ),
    # Government vs Opposition
    party_type = case_when(
      party_name %in% c("ODS", "STAN", "TOP 09", "KDU-ČSL") ~ "Government",
      party_name %in% c("ANO", "SPD") ~ "Opposition",
      TRUE ~ "Other"
    )
  )

# FIX: Safe word count function (vectorized, handles empty strings and NA)
count_words <- function(x) {
  ifelse(is.na(x) | str_squish(x) == "", 0L, str_count(str_squish(x), "\\S+"))
}

# Add word counts using vectorized function
data_final <- data_final %>%
  mutate(
    ingroup_word_count = count_words(ingroup_lemma),
    outgroup_word_count = count_words(outgroup_lemma)
  )

cat("Party distribution:\n")
print(table(data_final$party_name))

cat("\nGovernment/Opposition distribution:\n")
print(table(data_final$party_type))

cat("\nWord count statistics (lemmatized):\n")
cat("  Ingroup - Median:", median(data_final$ingroup_word_count),
    "Mean:", round(mean(data_final$ingroup_word_count), 1), "\n")
cat("  Outgroup - Median:", median(data_final$outgroup_word_count),
    "Mean:", round(mean(data_final$outgroup_word_count), 1), "\n")

# =============================================================================
# STEP 7: SAVE OUTPUT - TWO VERSIONS
# =============================================================================

cat("\n", rep("=", 60), "\n")
cat("STEP 7: Saving output (two versions)\n")
cat(rep("=", 60), "\n")

# VERSION 1: With lemmatization (for STM, keyness, DFM-MDS)
output_lemma <- paste0(OUTPUT_DIR, "czech_clean_lemmatized.csv")
write_csv(data_final, output_lemma)
cat("Saved:", output_lemma, "\n")

# VERSION 2: Original text only (for sentence transformers, BERTopic)
# Just clean the original text minimally - no lemmatization
data_original <- data_final %>%
  mutate(
    # Clean original text (minimal - just whitespace and basic cleaning)
    ingroup_clean = str_squish(open_ingroup),
    outgroup_clean = str_squish(open_outgroup)
  ) %>%
  select(
    respondent_id,
    # Original text (cleaned)
    ingroup_text = ingroup_clean,
    outgroup_text = outgroup_clean,
    # Keep lemmatized for reference
    ingroup_lemma,
    outgroup_lemma,
    # Metadata
    party_name, party_type, party_choice,
    ingroup_word_count, outgroup_word_count,
    # Keep all other columns
    everything()
  )

output_original <- paste0(OUTPUT_DIR, "czech_clean_original.csv")
write_csv(data_original, output_original)
cat("Saved:", output_original, "\n")

# =============================================================================
# SUMMARY
# =============================================================================

cat("\n", rep("=", 60), "\n")
cat("PREPROCESSING COMPLETE\n")
cat(rep("=", 60), "\n")

cat("\nPIPELINE SUMMARY:\n")
cat("  Raw data:              ", nrow(data_raw), "respondents\n")
cat("  After initial filter:  ", nrow(data_filtered), "respondents\n")
cat("  After 'don't know':    ", nrow(data_clean), "respondents\n")
cat("  Final clean data:      ", nrow(data_final), "respondents\n")
cat("  Overall retention:     ", round(nrow(data_final)/nrow(data_raw)*100, 1), "%\n")

cat("\nEXCLUSION SUMMARY (for paper):\n")
cat("  Missing responses:     ", nrow(data_raw) - nrow(data_filtered),
    " (", round((nrow(data_raw) - nrow(data_filtered))/nrow(data_raw)*100, 1), "%)\n")
cat("  'Don't know' responses:", length(exclude_either),
    " (", round(length(exclude_either)/nrow(data_filtered)*100, 1), "%)\n")
cat("\nNote: Short responses are NOT filtered here.\n")
cat("      Each analysis method handles this appropriately.\n")

cat("\nOUTPUT FILES:\n")
cat("  1. czech_clean_lemmatized.csv - Use for:\n")
cat("     - Keyness analysis (R)\n")
cat("     - STM topic modeling (R)\n")
cat("     - DFM + MDS bag-of-words (R/Python)\n")
cat("\n")
cat("  2. czech_clean_original.csv - Use for:\n")
cat("     - Sentence Transformer + MDS (Python)\n")
cat("     - BERTopic (Python)\n")
cat("     - Any neural language model analysis\n")
cat("\n")
cat("KEY COLUMNS:\n")
cat("  - respondent_id: unique identifier\n")
cat("  - ingroup_text, outgroup_text: original text (cleaned)\n")
cat("  - ingroup_lemma, outgroup_lemma: lemmatized text\n")
cat("  - party_name, party_type: political variables\n")

cat("\n", rep("=", 60), "\n")
