# GSE201753 nested cross-validation

This directory contains the code for the leakage-controlled nested cross-validation analysis of GSE201753.

## Design

- Binary outcome: ACT, RT, and RNT are combined as GCA; HD is Control.
- Samples: 109 total, including 80 GCA and 29 Control.
- Validation: five stratified outer folds and three stratified inner folds.
- Feature counts evaluated: 10, 25, and 50.
- Feature-selection methods: Lasso, SVM-RFE, Random Forest, SVM-RBF, Double-RBF, ReliefF, SigFeature, and Ensemble.
- Classifiers: Logistic Regression, Random Forest, and linear SVM.
- Selection criterion: mean inner-fold balanced accuracy, with macro-F1 used as the tie-breaker.
- Random seed: 42.

Low-count filtering, binary DESeq2, adjusted-P filtering, variance filtering, feature selection, scaling, SMOTE, and classifier fitting are repeated using the applicable training partition. Held-out samples do not determine preprocessing parameters, genes, hyperparameters, or model fitting.

DESeq2 uses the design `~ binary_class` and the contrast GCA versus Control. Positive log2 fold changes indicate higher expression in the combined GCA group; negative values indicate higher expression in HD controls.

## Requirements

Install the Python packages listed in the repository-level `requirements.txt`. R must be installed with `Rscript` available on `PATH`, and the Bioconductor package `DESeq2` must be installed.

## Run

From the repository root:

```bash
python src/nested_cv/run_GSE201753_nested_binary_cv.py
```

The runner reads:

- `data/GSE201753/GSE201753_raw_counts_GRCh38.p13_NCBI.tsv`
- `data/GSE201753/GSE201753_binary_metadata_109_samples.csv`

It writes results to `results/nested_cv`. If `results/nested_cv/preprocessed_splits/preprocessing_summary.csv` already exists, the runner reuses the existing fold-specific preprocessing artifacts.

## Saved results

- `nested_cv_summary.json`: pooled held-out metrics and confusion matrix.
- `outer_fold_results.csv`: selected configuration and held-out metrics for every outer fold.
- `outer_fold_predictions.csv`: one held-out prediction for each of the 109 samples.
- `outer_fold_selected_genes_with_symbols.csv`: fold-specific genes with Entrez IDs, symbols, descriptions, and selection frequencies.
- `inner_cv_all_configurations.csv`: all evaluated inner-CV configurations.
- `split_manifest.csv`: sample membership for every outer and inner training/validation split.
- `preprocessing_summary.csv`: summary of fold-specific DESeq2 preprocessing.

The pooled balanced accuracy, ROC AUC, sensitivity, and specificity represent the complete adaptive model-selection procedure. Different pipelines and gene panels were selected in different outer folds; these results are not an ensemble estimate or an estimate for one fixed final model.
