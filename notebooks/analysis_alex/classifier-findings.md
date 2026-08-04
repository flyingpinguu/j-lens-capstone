# Leakage classifier findings

This note summarizes the classifier experiments behind the self-contained
[`leakage-position-multilayer-classifier.ipynb`](leakage-position-multilayer-classifier.ipynb).
All reported
cross-validation uses `GroupKFold` with `category` as the group. The `control`
rows are treated as one ordinary held-out category group. Results therefore
measure transfer to unseen prompt families rather than random rows from known
families.

The dataset contains 1,320 runs: 924 attack prompts and 396 benign controls.
The original labels contain 458 leaks (34.7%); the corrected-label file changes
11 false negatives to leaks, giving 469 leaks (35.5%). Earlier results retain
their original-label values unless explicitly marked as corrected-label runs.
Lens readouts cover all 32 transformer layers and the user message, prompt
suffix/scaffolding and response. The primary prompt classifiers stop before
`response_start_position`; later response-only and stacking experiments are
explicitly separated because they contain post-outcome information.

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
- **XGBoost baseline:** the fold-local ten most frequent token IDs, kept
  separately for each of the 16 layers — 160 token-logit features and **no
  secret-rank features**.

`secret_rank` is the exact rank of the single-token secret (`banana`) at a
layer and position. The Logistic Regression still log-transforms it because
raw ranks are extremely right-skewed. The current per-position XGBoost never
reads it. Generic rank-1 through rank-10 logit values were also removed: they
added little signal without identifying which token received the logit.

The simple rank-free XGBoost baseline reached:

| Position | Features | Mean fold ROC-AUC | Standard deviation | OOF accuracy | OOF balanced accuracy |
|---|---:|---:|---:|---:|---:|
| suffix 5 | 160 | 0.762 | 0.050 | 60.8% | 63.1% |
| last prompt | 160 | 0.740 | 0.048 | 69.2% | 67.8% |
| suffix 3 | 160 | 0.739 | 0.043 | 69.1% | 70.6% |
| suffix 6 | 160 | 0.737 | 0.102 | 65.5% | 67.7% |

The `suffix_5` result still illustrates why ROC-AUC and accuracy must not be
confused: its ranking is strong while its fixed 0.5 threshold is poorly
calibrated. AUC measures ranking over every possible threshold; accuracy
depends on one chosen threshold and class prevalence.

Because ten frequency-selected tokens is an arbitrary design, the optional
search section in the self-contained classifier notebook compares vocabulary
widths, label association, frequency/association mixtures, SVD widths and
multiple linear/nonlinear classifier families. Its nested grouped-CV pipeline
chooses the representation and classifier inside every outer training fold.

The preferred one-standard-error nested results are:

| Position | Mean selected features | Mean outer-fold ROC-AUC | Standard deviation |
|---|---:|---:|---:|
| suffix 3 | 123 | 0.734 | 0.074 |
| suffix 4 | 32 | 0.732 | 0.048 |
| suffix 5 | 64 | 0.730 | 0.029 |
| last prompt | 101 | 0.724 | 0.055 |
| suffix 6 | 69 | 0.723 | 0.051 |

This nested estimate is more defensible after feature/model search than the
higher exploratory tuned scores. At suffix 4, all outer folds independently
selected 32 SVD components with shallow trees. Across scaffolding positions,
the unconstrained inner-CV winner averaged AUC 0.706 with about 473 features,
versus 0.705 with 160 features for the fixed frequency-10 baseline; the extra
complexity is not worthwhile as a global replacement.

#### Rank-free multi-model search

The XGBoost-only search was then broadened without changing the scientific
unit: there is still **one model per token position**, and that model sees the
Top-10 readouts from layers 16–31 jointly. The broader screen compared
regularized Logistic Regression, elastic-net Logistic Regression, linear and
RBF SVMs, Extra Trees and histogram gradient boosting. Candidate inputs
included raw logits, binary token presence, within-Top-10 softmax weights,
frequency/association mixtures, exact layer×token selection and 8/16/32/64
SVD components. No candidate reads `secret_rank`, probe fields, category or
response positions.

The most stable improvement is at `prompt_suffix_4`, the fixed `assistant`
chat-template token before generation. In all six outer folds, the inner
grouped CV independently chose the same pipeline:

1. select the 100 most frequent Top-10 token IDs in the training fold;
2. form 1,600 named layer×token logit columns (`16 layers × 100 tokens`);
3. fit a 16-component Truncated SVD on that training matrix;
4. fit class-balanced L2 Logistic Regression (`C=0.1`) on those 16 features.

The leading nested one-standard-error results are:

| Position | Mean selected features | Mean outer-fold ROC-AUC | Standard deviation | OOF accuracy | OOF balanced accuracy |
|---|---:|---:|---:|---:|---:|
| suffix 4 (`assistant`) | 16 | 0.771 | 0.043 | 69.5% | 71.4% |
| suffix 6 | 20 | 0.738 | 0.058 | 64.2% | 66.8% |
| suffix 5 | 48 | 0.722 | 0.056 | 51.1% | 55.8% |
| suffix 7 | 29 | 0.717 | 0.059 | 60.9% | 63.7% |
| suffix 3 | 32 | 0.714 | 0.050 | 56.7% | 61.1% |

For suffix 4 this is a 0.039 AUC improvement over the nested XGBoost result
(0.732) with half as many final features (16 instead of 32). It also beats
the 65.3% majority-class accuracy baseline. The gain is not universal:
XGBoost remains slightly better at suffix 3 and suffix 5. This is why the
notebook retains the full per-position comparison instead of claiming that
one classifier family dominates everywhere.

The 16 SVD columns are dense latent components, not 16 selected tokens. Each
is a train-fold-fitted linear combination of the original named layer×token
columns. This avoids thousands of sparse classifier coefficients while still
preserving token identity in the matrix being compressed. The notebook shows
both an actual 16-column feature-matrix slice and the layer/token columns with
the largest loadings in each component.

Rerunning the locked suffix-4 pipeline on the corrected labels gives mean fold
ROC-AUC 0.778, accuracy 69.4% and balanced accuracy 71.3%. The correction
therefore does not change the prompt-only conclusion.

### Response-position classifiers (post-outcome)

As a deliberately separate experiment, generated responses were divided into
ten non-overlapping relative bins (`0–10%`, ..., `90–100%`). Within each bin
and layer, repeated Top-10 token IDs were max-pooled and the ten strongest
token logits retained. Every bin model then used the same rank-free pipeline:

1. layers 16–31 at that response region;
2. a training-fold top-100 token vocabulary and 1,600 layer×token columns;
3. 16 fold-local SVD components;
4. either class-balanced Logistic Regression or XGBoost.

The corrected-label results include:

| Response region | Logistic ROC-AUC | XGBoost ROC-AUC |
|---|---:|---:|
| 0–10% | 0.637 | 0.722 |
| 10–20% | 0.762 | 0.770 |
| 20–30% | 0.597 | 0.629 |
| 90–100% | 0.637 | 0.669 |

The strongest pooled region is 10–20%. Exact single-token checks show that
XGBoost reaches AUC 0.690 at 10%, 0.698 at 15% and 0.570 at 20%. Pooling the
10–20% span is therefore more informative than any tested individual token;
the detectable response signal appears distributed across several neighboring
generated tokens.

These results have substantial category-fold variability and are inherently
outcome-circular: leaked and resisted responses already contain different
generated vocabulary, and the ordinary Top 10 may contain the secret itself.
They describe post-response leakage detection, not prediction before the
response is generated.

### Nested prompt/response stacking (post-outcome)

The stacking experiment used ten base scores:

- prompt suffix 4 (`assistant`) and suffix 6 (`<think>`), each modeled by
  Logistic Regression and XGBoost;
- response bins 0–10%, 10–20% and 90–100%, also modeled by both algorithms.

Evaluation used proper nested cross-fitting. Within every outer
`GroupKFold(category)` split, four inner category folds generated the
meta-training predictions. Each base model was then refitted using only the
outer-training categories before predicting the untouched outer categories.
Thus no outer test label enters the base or meta-model training process.

| Model | Mean outer-fold ROC-AUC |
|---|---:|
| Best prompt base: suffix-4 Logistic | 0.778 |
| Best response base: 10–20% XGBoost | 0.770 |
| Best prompt-only stack | 0.779 |
| Best response-only stack | 0.777 |
| All ten prompt/response scores | 0.832 |
| **Compact four-score prompt/response stack** | **0.847 ± 0.047** |

The compact stack combines suffix-4 Logistic/XGBoost with response-10–20%
Logistic/XGBoost, using shallow XGBoost as the meta-model. It reaches pooled
OOF AUC 0.788, accuracy 73.4% and balanced accuracy 71.8%. The difference
between mean fold AUC and pooled OOF AUC indicates category-dependent score
calibration: ranking is stronger within held-out folds than after scores from
all folds are pooled on one global scale.

The key exploratory takeaway is that stacking Logistic Regression and
XGBoost within one modality adds almost nothing; the useful complementarity
comes from combining prompt and response views. The compact four-score stack
also beats the more correlated ten-score stack. However, 18 stack
configurations were compared on the same six outer folds, so 0.847 is a
model-selection estimate rather than an untouched final-test result. Because
response features are included, this remains a post-response detector and
cannot replace the prompt-only pre-response analysis.

## Feature-engineering lessons

1. Token identity matters. Logit values without their associated token added
   little compared with sparse token-aware features.
2. Position identity matters. Keeping coarse user regions and exact suffix
   positions was better than globally max-pooling the whole prompt.
3. Compact representations generalized at least as well as thousands of
   token columns. Per-position XGBoost uses 160 rank-free baseline columns,
   while the strongest suffix-4 model uses only 16 SVD components.
4. Vocabulary selection and dimensionality reduction must be fitted inside
   each training fold. Otherwise held-out prompt families influence the
   feature space.
5. `secret_rank` is useful but secret-specific. Removing it from per-position
   XGBoost retains useful suffix-position AUC around 0.72–0.73 under nested
   evaluation, so that signal is not solely the banana probe.
6. Category is never used as a predictor. It exists only for grouped
   validation.
7. Direct label-correlation selection is not automatically better. It can
   select category-specific vocabulary; fold-local SVD and small mixtures were
   more stable, and no one representation dominated every position.

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
