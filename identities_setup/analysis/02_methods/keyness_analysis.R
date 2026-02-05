# =============================================================================
# KEYNESS ANALYSIS FOR CZECH POLITICAL IDENTITY DATA
# =============================================================================
#
# This script runs keyness analysis to identify distinctive words:
# 1. Ingroup vs Outgroup (overall)
# 2. Government vs Opposition voters (for both ingroup and outgroup)
#
# Requires: czech_clean_for_analysis.csv (from preprocessing script)
#
# Author: [Your name]
# =============================================================================

library(readr)
library(dplyr)
library(stringr)
library(quanteda)
library(quanteda.textstats)
library(quanteda.textplots)
library(stopwords)
library(ggplot2)

# =============================================================================
# LOAD PREPROCESSED DATA
# =============================================================================

cat("Loading preprocessed data...\n")

# Try multiple paths
if (file.exists("czech_clean_for_analysis.csv")) {
  data <- read_csv("czech_clean_for_analysis.csv")
} else if (file.exists("/Users/lenkahrbkova/Downloads/czech_clean_for_analysis.csv")) {
  data <- read_csv("/Users/lenkahrbkova/Downloads/czech_clean_for_analysis.csv")
} else {
  stop("Cannot find czech_clean_for_analysis.csv - run preprocessing first!")
}

cat("Loaded", nrow(data), "respondents\n")

# Czech stopwords
custom_stopwords <- c(stopwords("cs", source = "stopwords-iso"), "pan", "paní")

# =============================================================================
# ANALYSIS 1: INGROUP VS OUTGROUP (OVERALL)
# =============================================================================

cat("\n", rep("=", 60), "\n")
cat("ANALYSIS 1: INGROUP vs OUTGROUP KEYNESS\n")
cat(rep("=", 60), "\n")

# Create corpora
corpus_ingroup <- corpus(data$ingroup_lemma)
corpus_outgroup <- corpus(data$outgroup_lemma)

# Tokenize
tokens_ingroup <- tokens(corpus_ingroup, remove_punct = TRUE, remove_numbers = TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(custom_stopwords)

tokens_outgroup <- tokens(corpus_outgroup, remove_punct = TRUE, remove_numbers = TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(custom_stopwords)

# Create DFMs
dfm_ingroup <- dfm(tokens_ingroup)
dfm_outgroup <- dfm(tokens_outgroup)

# Combine and add group labels
dfm_combined <- rbind(dfm_ingroup, dfm_outgroup)
docvars(dfm_combined, "group") <- c(rep("ingroup", ndoc(dfm_ingroup)),
                                     rep("outgroup", ndoc(dfm_outgroup)))

# Calculate keyness
keyness_inout <- textstat_keyness(dfm_combined,
                                   target = docvars(dfm_combined, "group") == "ingroup")

cat("\nTOP 20 WORDS MORE ASSOCIATED WITH INGROUPS:\n")
print(head(keyness_inout, 20))

cat("\nTOP 20 WORDS MORE ASSOCIATED WITH OUTGROUPS:\n")
print(tail(keyness_inout, 20))

# Save results
write_csv(keyness_inout, "keyness_ingroup_vs_outgroup.csv")

# =============================================================================
# ANALYSIS 2: GOVERNMENT VS OPPOSITION - INGROUPS
# =============================================================================

cat("\n", rep("=", 60), "\n")
cat("ANALYSIS 2: GOVERNMENT vs OPPOSITION (INGROUPS)\n")
cat(rep("=", 60), "\n")

# Filter to Government and Opposition only
partisan_data <- data %>%
  filter(party_type %in% c("Government", "Opposition"))

cat("Partisan sample:", nrow(partisan_data), "respondents\n")
cat("Government:", sum(partisan_data$party_type == "Government"), "\n")
cat("Opposition:", sum(partisan_data$party_type == "Opposition"), "\n")

# Create tokens by party type
gov_ingroup_tokens <- tokens(partisan_data$ingroup_lemma[partisan_data$party_type == "Government"]) %>%
  tokens_remove(custom_stopwords)
opp_ingroup_tokens <- tokens(partisan_data$ingroup_lemma[partisan_data$party_type == "Opposition"]) %>%
  tokens_remove(custom_stopwords)

# Create DFMs
gov_ingroup_dfm <- dfm(gov_ingroup_tokens)
opp_ingroup_dfm <- dfm(opp_ingroup_tokens)

# Keyness analysis
ingroup_keyness <- textstat_keyness(rbind(gov_ingroup_dfm, opp_ingroup_dfm),
                                     target = seq_len(ndoc(gov_ingroup_dfm)))

cat("\nTOP 20 GOVERNMENT-DISTINCTIVE INGROUP WORDS:\n")
print(head(ingroup_keyness, 20))

cat("\nTOP 20 OPPOSITION-DISTINCTIVE INGROUP WORDS:\n")
print(tail(ingroup_keyness, 20))

write_csv(ingroup_keyness, "keyness_gov_vs_opp_ingroup.csv")

# =============================================================================
# ANALYSIS 3: GOVERNMENT VS OPPOSITION - OUTGROUPS
# =============================================================================

cat("\n", rep("=", 60), "\n")
cat("ANALYSIS 3: GOVERNMENT vs OPPOSITION (OUTGROUPS)\n")
cat(rep("=", 60), "\n")

# Create tokens by party type for outgroups
gov_outgroup_tokens <- tokens(partisan_data$outgroup_lemma[partisan_data$party_type == "Government"]) %>%
  tokens_remove(custom_stopwords)
opp_outgroup_tokens <- tokens(partisan_data$outgroup_lemma[partisan_data$party_type == "Opposition"]) %>%
  tokens_remove(custom_stopwords)

# Create DFMs
gov_outgroup_dfm <- dfm(gov_outgroup_tokens)
opp_outgroup_dfm <- dfm(opp_outgroup_tokens)

# Keyness analysis
outgroup_keyness <- textstat_keyness(rbind(gov_outgroup_dfm, opp_outgroup_dfm),
                                      target = seq_len(ndoc(gov_outgroup_dfm)))

cat("\nTOP 20 GOVERNMENT-DISTINCTIVE OUTGROUP WORDS:\n")
print(head(outgroup_keyness, 20))

cat("\nTOP 20 OPPOSITION-DISTINCTIVE OUTGROUP WORDS:\n")
print(tail(outgroup_keyness, 20))

write_csv(outgroup_keyness, "keyness_gov_vs_opp_outgroup.csv")

# =============================================================================
# TRANSLATIONS FOR PLOTTING
# =============================================================================

# Translation dictionaries (expand as needed)
ingroup_translations <- data.frame(
  czech = c("stejný", "střední", "normální", "podobný", "rozumný", "pracující",
            "svoboda", "spravedlnost", "slušný", "rodina", "demokracie", "ověřovat",
            "inteligentní", "informace", "vzdělaný", "zodpovědnost", "selský",
            "vláda", "občan", "obyčejný"),
  english = c("same", "middle", "normal", "similar", "reasonable", "working",
              "freedom", "justice", "decent", "family", "democracy", "to verify",
              "intelligent", "information", "educated", "responsibility", "common sense",
              "government", "citizen", "ordinary")
)

outgroup_translations <- data.frame(
  czech = c("názor", "jiný", "hloupý", "bohatý", "dezinformace", "sobecký",
            "spd", "moci", "nízký", "pravda", "nadávat", "osoba", "právo",
            "výhoda", "volit", "zisk", "myslit", "prospěch", "věřit", "populista"),
  english = c("opinion", "different", "stupid", "rich", "disinformation", "selfish",
              "SPD", "power", "low", "truth", "to curse", "person", "right",
              "advantage", "to vote", "profit", "to think", "benefit", "to believe", "populist")
)

# =============================================================================
# PLOTTING FUNCTION
# =============================================================================

create_keyness_plot <- function(keyness_data, translations, title, n_words = 20) {

  # Get top and bottom words
  top_words <- rbind(head(keyness_data, n_words), tail(keyness_data, n_words))

  # Add translations
  top_words <- merge(top_words, translations,
                     by.x = "feature", by.y = "czech", all.x = TRUE)

  # Create display names
  top_words$display_name <- ifelse(is.na(top_words$english),
                                    top_words$feature,
                                    top_words$english)

  # Create group labels
  top_words$group <- ifelse(top_words$chi2 > 0, "Target", "Reference")

  # Plot
  p <- ggplot(top_words, aes(x = chi2, y = reorder(display_name, chi2), fill = group)) +
    geom_col() +
    scale_fill_manual(values = c("Target" = "#2c5282", "Reference" = "#8c8c8c")) +
    labs(title = title, x = "Chi-squared statistic", y = "") +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      legend.title = element_blank(),
      plot.title = element_text(hjust = 0.5, face = "bold")
    ) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50")

  return(p)
}

# =============================================================================
# CREATE AND SAVE PLOTS
# =============================================================================

cat("\n", rep("=", 60), "\n")
cat("CREATING PLOTS\n")
cat(rep("=", 60), "\n")

# Plot 1: Ingroup vs Outgroup
all_translations <- rbind(ingroup_translations, outgroup_translations) %>% distinct()

p1 <- create_keyness_plot(keyness_inout, all_translations,
                          "Keyness: Ingroup vs Outgroup Descriptions")
ggsave("keyness_ingroup_vs_outgroup.png", p1, width = 12, height = 8, dpi = 300)
cat("Saved: keyness_ingroup_vs_outgroup.png\n")

# Plot 2: Gov vs Opp Ingroups
p2 <- create_keyness_plot(ingroup_keyness, all_translations,
                          "Keyness: Government vs Opposition (Ingroup Descriptions)")
ggsave("keyness_gov_opp_ingroup.png", p2, width = 12, height = 8, dpi = 300)
cat("Saved: keyness_gov_opp_ingroup.png\n")

# Plot 3: Gov vs Opp Outgroups
p3 <- create_keyness_plot(outgroup_keyness, all_translations,
                          "Keyness: Government vs Opposition (Outgroup Descriptions)")
ggsave("keyness_gov_opp_outgroup.png", p3, width = 12, height = 8, dpi = 300)
cat("Saved: keyness_gov_opp_outgroup.png\n")

# =============================================================================
# SUMMARY
# =============================================================================

cat("\n", rep("=", 60), "\n")
cat("KEYNESS ANALYSIS COMPLETE\n")
cat(rep("=", 60), "\n")

cat("\nFILES CREATED:\n")
cat("  - keyness_ingroup_vs_outgroup.csv\n")
cat("  - keyness_gov_vs_opp_ingroup.csv\n")
cat("  - keyness_gov_vs_opp_outgroup.csv\n")
cat("  - keyness_ingroup_vs_outgroup.png\n")
cat("  - keyness_gov_opp_ingroup.png\n")
cat("  - keyness_gov_opp_outgroup.png\n")

cat("\nKEY FINDINGS:\n")
cat("Ingroup distinctive words suggest how people define 'us'\n")
cat("Outgroup distinctive words suggest how people define 'them'\n")
cat("Government vs Opposition comparison reveals partisan framing differences\n")
