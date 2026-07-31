# Leakage classifier findings

This note summarizes the classifier experiments that preceded the consolidated
[`leakage-classifiers.ipynb`](leakage-classifiers.ipynb). All reported
cross-validation uses `GroupKFold` with `category` as the group. The `control`
rows are treated as one ordinary held-out category group. Results therefore
measure transfer to unseen prompt families rather than random rows from known
families.

The dataset contains 1,320 runs: 924 attack prompts, 396 benign controls and
458 leaks (34.7%). Lens readouts cover all 32 transformer layers and the user
message, prompt suffix/scaffolding and response. Classifier inputs stop before
`response_start_position`, so no response token or post-outcome information is
used.

## What worked best

### One classifier per transformer layer

The strongest layer-local representation was a sparse
**position-group × token-logit** matrix:

- user-message tokens are max-pooled within five relative position bins;
- the nine exact prompt-suffix positions remain separate;
- within each cross-validation fold, the 300 most frequent top-readout tokens
  are selected using training rows only;
- a feature is the maximum logit for one selected token in one position group;
- missing token/group combinations remain sparse/missing;
- category is only a fold group and is never a model feature.

This produces about 4,200 candidate columns per layer (`14 × 300`), although
each prompt activates only a small fraction. Models are fitted separately for
each transformer layer; the curve across layers is the localization result.

Best settings retained:

- logistic regression: `C=0.03`;
- XGBoost: 400 trees, depth 3, learning rate 0.03 and minimum child weight 3.

Best mean fold ROC-AUC:

| Model | Best layer | ROC-AUC | Standard deviation |
|---|---:|---:|---:|
| Logistic regression | 9 | 0.729 | 0.043 |
| XGBoost | 26 | 0.786 | 0.021 |

Layer 31 reached 0.713 with logistic regression and 0.779 with XGBoost.
Because residual-stream information is correlated across neighboring layers,
these independent per-layer models are more interpretable for localization
than one joint model over all layers.

### One classifier per token position

The retained position-local analyses use transformer layers 16–31. Positions
are ten relative user-message bins, the exact last user token, eight prompt
suffix positions and the exact final prompt position. A separate classifier
is trained for every position definition.

The most useful compact feature sets differed by model:

- **Logistic regression:** a fold-local top-100 layer×token matrix compressed
  to 32 Truncated-SVD components, plus 16 `log(secret_rank)` values — about 48
  features.
- **XGBoost:** the fold-local ten most frequent token IDs, kept separately for
  each of the 16 layers, plus 16 `log(secret_rank)` values — 176 features.

`secret_rank` is the exact rank of the single-token secret (`banana`) at a
layer and position. It is log-transformed because raw ranks are extremely
right-skewed. Generic rank-1 through rank-10 logit values were removed: they
added little signal without identifying which token received the logit.

Best mean fold ROC-AUC:

| Model | Position | Features | ROC-AUC | Standard deviation |
|---|---|---:|---:|---:|
| Logistic regression | suffix 4 | 48 | 0.757 | 0.015 |
| XGBoost | suffix 5 | 176 | 0.761 | 0.057 |
| XGBoost | last prompt token | 176 | 0.758 | 0.057 |

The XGBoost `suffix_5` result illustrates why ROC-AUC and accuracy must not be
confused. It has AUC 0.761 but out-of-fold accuracy 60.2% at the fixed 0.5
threshold and balanced accuracy 62.4%. The `last_prompt` model has similar AUC
(0.758) but 69.8% accuracy and 68.9% balanced accuracy. AUC measures ranking
over every possible threshold; accuracy depends on one chosen threshold and
class prevalence.

## Feature-engineering lessons

1. Token identity matters. Logit values without their associated token added
   little compared with sparse token-aware features.
2. Position identity matters. Keeping coarse user regions and exact suffix
   positions was better than globally max-pooling the whole prompt.
3. Compact representations generalized at least as well as thousands of
   token columns. For per-position XGBoost, reducing roughly 5,000 token
   features to 176 improved the result.
4. Vocabulary selection and dimensionality reduction must be fitted inside
   each training fold. Otherwise held-out prompt families influence the
   feature space.
5. `secret_rank` is useful but secret-specific. The token-logit features test
   whether leakage is predictable from broader readout vocabulary; the rank
   features provide a small targeted companion signal.
6. Category is never used as a predictor. It exists only for grouped
   validation.

## Joint late-prompt classifier

The consolidated notebook also evaluates one classifier design that combines
all user-message positions from 80% onward, the exact last user token and all
nine suffix/scaffolding positions across layers 16–31. Response positions
remain excluded.

The best tested variant was XGBoost with the standard settings (400 trees,
depth 3, learning rate 0.03 and minimum child weight 3). Within every training
fold it:

- starts from the 50 most frequent tokens;
- keeps token logits distinct for all 11 position groups and 16 layers;
- selects the 64 token-logit columns with the strongest training-fold
  ANOVA F-scores;
- appends 176 `log(secret_rank)` features (`11 groups × 16 layers`).

The resulting 240-feature model reached:

| Mean fold ROC-AUC | Fold standard deviation | OOF ROC-AUC | OOF accuracy | OOF balanced accuracy |
|---:|---:|---:|---:|---:|
| 0.767 | 0.027 | 0.763 | 71.7% | 70.9% |

For context, the strongest compact alternatives were:

| Representation | Features | Mean fold ROC-AUC |
|---|---:|---:|
| Position×layer supervised token features + ranks | 240 | 0.767 |
| Globally pooled top-25 token features + ranks | 201 | 0.764 |
| Position×layer SVD-64 token features + ranks | 240 | 0.761 |
| Secret ranks only | 176 | 0.755 |

These comparisons are exploratory: feature representations and XGBoost
settings were compared on the same grouped cross-validation folds used to
quote their scores. The 0.767 result is therefore a selection estimate, not
an untouched final-test estimate. Vocabulary selection, F-score selection and
SVD fitting themselves were still performed strictly inside each training
fold.

These joint features combine correlated positions and layers. Their result is
a predictive late-prompt summary and must not be interpreted as locating one
causal layer or one point where the model “decides” to leak.
