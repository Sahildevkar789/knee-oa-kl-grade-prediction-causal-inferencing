# Ensemble (Part A) summary

## Selection steps (validation)

|   step | added          | members                              |    acc |    qwk |     f1 |    mae |
|-------:|:---------------|:-------------------------------------|-------:|-------:|-------:|-------:|
|      0 | kl_final_seed7 | ['kl_final_seed7']                   | 0.6889 | 0.8406 | 0.6975 | 0.362  |
|      1 | kneenext_seed7 | ['kl_final_seed7', 'kneenext_seed7'] | 0.6925 | 0.8491 | 0.7103 | 0.3475 |

Threshold tuning: adopted (5-fold CV gain +0.0061). Final thresholds: [0.7, 0.45, 0.5, 0.55].

Flip TTA: not adopted (val acc 0.7107 -> 0.6985).

Selected models: ['kl_final_seed7', 'kneenext_seed7']

## Test set (evaluated once)

Accuracy **0.7172**, QWK **0.8574**, Macro F1 **0.6933**, MAE **0.3293**.

Above the 70% accuracy target.

### vs kl_final

| Metric                         |   MeanDiff |   CI_low |   CI_high | Significant_95pct   |   Resamples |
|:-------------------------------|-----------:|---------:|----------:|:--------------------|------------:|
| QWK (ensemble - kl_final)      |     0.0005 |  -0.009  |    0.0096 | False               |        2000 |
| Accuracy (ensemble - kl_final) |     0.0234 |   0.0073 |    0.0405 | True                |        2000 |

### vs baseline

| Metric                         |   MeanDiff |   CI_low |   CI_high | Significant_95pct   |   Resamples |
|:-------------------------------|-----------:|---------:|----------:|:--------------------|------------:|
| QWK (ensemble - baseline)      |     0.0329 |   0.0162 |    0.0496 | True                |        2000 |
| Accuracy (ensemble - baseline) |     0.0886 |   0.0616 |    0.1148 | True                |        2000 |

