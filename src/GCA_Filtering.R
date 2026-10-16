library(DESeq2)

# Load RAW counts (not TPM-normalized) — rows = genes, columns = samples
counts <- read.csv("/Users/damianmoreno/Documents/DePaul_Cobab/GCA_Project/GSE224526/GSE224526_raw_counts_GRCh38.p13_NCBI.tsv", row.names = 1, sep = "\t")

# Load metadata — must have one row per sample, with a 'condition' column
metadata <- read.csv("/Users/damianmoreno/Documents/DePaul_Cobab/GCA_Project/GSE224526/GSE224526_metadata.csv", row.names = 1)

# Ensure sample order matches between counts matrix and metadata
print(length(colnames(counts)))
print(length(rownames(metadata)))
counts <- counts[, rownames(metadata)]

# Round to integers just in case (DESeq2 requires integer counts)
counts <- round(counts)

# Build DESeqDataSet
dds <- DESeqDataSetFromMatrix(
    countData = counts,
    colData   = metadata,
    design    = ~ condition   # 'condition' column must exist in metadata
)

# Pre-filter: remove genes with very low total counts across all samples
# (speeds up DESeq2 and reduces noise)
keep <- rowSums(counts(dds)) >= 10
dds  <- dds[keep, ]

# Run DESeq2
dds <- DESeq(dds)

# Extract results with adjusted p-value threshold
res <- results(dds, alpha = 0.05)

# Filter to significant DEGs (padj < 0.05)
res_sig <- subset(res, padj < 0.05)
cat("Number of DEGs:", nrow(res_sig), "\n")

# Export the filtered normalized counts for significant genes
# (this is what you feed into the Python notebook)
normalized_counts <- counts(dds, normalized = TRUE)
deg_counts <- normalized_counts[rownames(res_sig), ]

write.csv(as.data.frame(deg_counts), "/Users/damianmoreno/Documents/DePaul_Cobab/GCA_Project/GSE224526/deg_filtered_counts.csv")

# Also export the full DESeq2 results table for reference
write.csv(as.data.frame(res_sig), "/Users/damianmoreno/Documents/DePaul_Cobab/GCA_Project/GSE224526/deseq2_results.csv")