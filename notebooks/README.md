# Optional exploration

The reproducible pipeline lives in `src/`; notebooks are not required to train or score.
For interactive analysis, load `outputs/training_landmarks.parquet`,
`outputs/test_predictions.csv`, or query `data/warehouse.duckdb`. Keep generated
notebooks out of the training path so execution order cannot change the results.
