suppressPackageStartupMessages(library(DESeq2))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Usage: Rscript nested_binary_deseq2_preprocess.R RAW_COUNTS METADATA SPLIT_MANIFEST OUTPUT_DIR")
}

raw_path <- args[[1]]
metadata_path <- args[[2]]
manifest_path <- args[[3]]
output_dir <- args[[4]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

counts_df <- read.delim(raw_path, header = TRUE, row.names = 1, check.names = FALSE)
counts_all <- round(as.matrix(counts_df))
storage.mode(counts_all) <- "integer"

metadata <- read.csv(metadata_path, stringsAsFactors = FALSE, check.names = FALSE)
rownames(metadata) <- metadata$sample_id
metadata$binary_class <- factor(metadata$binary_class, levels = c("Control", "GCA"))

manifest <- read.csv(manifest_path, stringsAsFactors = FALSE)
required <- c("split_id", "sample_id", "role")
if (!all(required %in% colnames(manifest))) stop("Split manifest is missing required columns")
if (!all(unique(manifest$role) %in% c("train", "validation"))) stop("Invalid split role")

summaries <- list()
for (split_id in unique(manifest$split_id)) {
  cat("Processing", split_id, "\n")
  rows <- manifest[manifest$split_id == split_id, , drop = FALSE]
  train_ids <- rows$sample_id[rows$role == "train"]
  valid_ids <- rows$sample_id[rows$role == "validation"]
  missing_ids <- setdiff(c(train_ids, valid_ids), colnames(counts_all))
  if (length(missing_ids)) stop("Samples missing from counts for ", split_id, ": ", paste(missing_ids, collapse = ", "))

  train_counts <- counts_all[, train_ids, drop = FALSE]
  valid_counts <- counts_all[, valid_ids, drop = FALSE]

  # Every data-derived decision below is learned from training samples only.
  keep <- rowSums(train_counts) >= 10
  train_counts <- train_counts[keep, , drop = FALSE]
  valid_counts <- valid_counts[keep, , drop = FALSE]

  train_meta <- metadata[train_ids, , drop = FALSE]
  dds <- DESeqDataSetFromMatrix(
    countData = train_counts,
    colData = train_meta,
    design = ~ binary_class
  )

  # Freeze the DESeq2 median-ratio reference on training data and use that
  # reference to normalize both training and validation samples.
  geo_means <- exp(rowMeans(log(as.matrix(train_counts))))
  geo_means[!is.finite(geo_means)] <- 0
  dds <- estimateSizeFactors(dds, geoMeans = geo_means)
  dds <- DESeq(dds, quiet = TRUE)
  res <- results(dds, contrast = c("binary_class", "GCA", "Control"), alpha = 0.05)
  res_df <- as.data.frame(res)
  res_df$gene_id <- rownames(res_df)
  res_df <- res_df[order(res_df$padj, na.last = TRUE), , drop = FALSE]
  sig <- res_df[!is.na(res_df$padj) & res_df$padj < 0.05, , drop = FALSE]
  genes <- sig$gene_id

  split_dir <- file.path(output_dir, split_id)
  dir.create(split_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(res_df, file.path(split_dir, "deseq2_results.csv"), row.names = FALSE)

  if (length(genes) < 2) {
    summaries[[split_id]] <- data.frame(
      split_id = split_id, n_train = length(train_ids), n_validation = length(valid_ids),
      genes_after_count_filter = nrow(train_counts), significant_genes = length(genes),
      variance_threshold = NA_real_, final_genes = 0, status = "insufficient_DEGs"
    )
    next
  }

  train_norm <- counts(dds, normalized = TRUE)[genes, , drop = FALSE]
  valid_sf <- estimateSizeFactorsForMatrix(valid_counts, geoMeans = geo_means)
  valid_norm <- sweep(valid_counts[genes, , drop = FALSE], 2, valid_sf, "/")
  train_expr <- log2(t(train_norm) + 1)
  valid_expr <- log2(t(valid_norm) + 1)

  train_var <- apply(train_expr, 2, var, na.rm = TRUE)
  variance_threshold <- as.numeric(quantile(train_var, probs = 0.20, na.rm = TRUE, names = FALSE))
  final_genes <- names(train_var)[is.finite(train_var) & train_var >= variance_threshold]
  train_expr <- train_expr[, final_genes, drop = FALSE]
  valid_expr <- valid_expr[, final_genes, drop = FALSE]

  train_out <- data.frame(sample_id = rownames(train_expr), binary_class = metadata[rownames(train_expr), "binary_class"], train_expr, check.names = FALSE)
  valid_out <- data.frame(sample_id = rownames(valid_expr), binary_class = metadata[rownames(valid_expr), "binary_class"], valid_expr, check.names = FALSE)
  write.csv(train_out, file.path(split_dir, "train_expression.csv"), row.names = FALSE)
  write.csv(valid_out, file.path(split_dir, "validation_expression.csv"), row.names = FALSE)

  summaries[[split_id]] <- data.frame(
    split_id = split_id, n_train = length(train_ids), n_validation = length(valid_ids),
    genes_after_count_filter = nrow(train_counts), significant_genes = length(genes),
    variance_threshold = variance_threshold, final_genes = length(final_genes), status = "ok"
  )
}

summary_df <- do.call(rbind, summaries)
write.csv(summary_df, file.path(output_dir, "preprocessing_summary.csv"), row.names = FALSE)
cat("Completed", nrow(summary_df), "training-only preprocessing splits.\n")
