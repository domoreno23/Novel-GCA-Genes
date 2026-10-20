# Novel GCA Genes
A feature selection methodology for discovering novel genes in Giant Cell Artritis. Relevant data and source code used in the study can be found here as well as images used in the paper.

## Leakage-controlled nested cross-validation

The reproducible five-outer-fold, three-inner-fold nested cross-validation analysis is documented under [`src/nested_cv`](src/nested_cv). Its pooled held-out predictions, fold-level results, selected genes, split manifest, and summary metrics are available under [`results/nested_cv`](results/nested_cv).

This analysis combines ACT, RT, and RNT into the GCA class and compares them with HD controls. DESeq2 preprocessing, feature selection, scaling, SMOTE, and classifier fitting are repeated using the applicable training partition. The pooled metrics describe the adaptive nested model-selection procedure; they are not the performance of an ensemble or one fixed 10-gene model.
