# RETRO — what the week taught

Six days, one project, the full machine-learning lifecycle. These are the lessons that stuck, per stage.

## Explore (Mon)

- **Look before you model.** Twenty minutes of groupby told us more than any algorithm did all week ("women 74% vs men 19%").
- **The dumb baseline is the real bar.** "All women survive" = 78.7%. Any model that can't beat a human-applied rule isn't adding value.
- The missing-value audit isn't housekeeping — it *is* modeling. 77% missing cabins became a `has_cabin` flag, which carried signal.

## Prepare (Tue)

- Feature engineering beat algorithm choice this week. `Title` (a sex×age interaction, pre-packaged) did more for the linear model than tuning did for the trees.
- A pipeline that welds preprocessing to the model is the difference between a notebook trick and a system — the API *cannot* transform data differently than training did.
- Imputation statistics fit on train only. Always. (Leakage lives in the boring steps.)

## Train (Tue)

- **The simplest model won the first round** — and that was a feature, not a bug: watching the decision tree hit 99% train / 78% val made overfitting *felt*, not defined.
- Experiment tracking isn't bureaucracy. MLflow's receipts are the only reason Wednesday's champion is reproducible.

## Judge (Wed)

- **One split is an anecdote; five folds are evidence.** CV overturned Tuesday's ranking (logistic regression → gradient boosting). Trusting Tuesday's number would have shipped the wrong champion.
- Accuracy hides the interesting failures. The confusion matrix confessed: recall 0.70 on survivors — 3 of 10 real survivors mislabeled dead. Decide which error is worse *for the problem*, not just the metric.
- **Tuning bought +0.7pt.** Hyperparameters squeeze; features and data pour.

## Ship (Thu)

- Notebooks are conversations; modules are contracts. Promotion wasn't copy-paste — the tests caught bugs the notebook tolerated (partial payloads, missing columns).
- TDD caught 3 real bugs on day one of trying it. Tests written first define the API before it exists.
- Docker is the "works on my machine" cure; CI is the bouncer. Both are habits, not heroics.

## Watch (Fri)

- **Monitor inputs, not just outputs.** Accuracy needs labels; labels arrive late. Feature distributions are knowable immediately — the smoke alarm, not the fire report.
- A drift detector that fires on everything is as useless as one that fires on nothing. Plant drift, verify it catches exactly that, verify a quiet month stays quiet.
- The registry (v1 → v2) makes retraining *versioned and roll-back-able* instead of overwriting a file and hoping.

## The meta-lesson

Every time my expectation was wrong (trees will win → they didn't; LR won → CV disagreed), the process had already been built to catch it. **The skill isn't knowing the answer — it's building the arena where evidence decides.**

---

# Math cheat sheet (one page)

**Logistic regression** — a weighted sum of features, squashed into a 0–1 probability by the sigmoid; trained by nudging weights downhill on the log-loss. The simplest classifier — and hard to beat on small data.

**Sigmoid** — σ(z) = 1/(1+e⁻ᶻ). Turns any number into a probability: 0 → 50%, big positive → ~100%, big negative → ~0%.

**Log-loss** — the training target: punished in proportion to how *confidently wrong* a prediction was. Confidently right ≈ free; confidently wrong ≈ very expensive.

**Gradient descent** — the learning loop: predict → measure loss → step every weight a little opposite the loss's steepest increase → repeat.

**Gradient boosting** — many small trees in sequence, each correcting the previous ensemble's errors, scaled by a learning rate. Flexible; finds curves and interactions; watch the train/val gap.

**Random forest** — many trees on random data subsets, majority vote. Averaging cancels individual trees' memorization.

**Entropy / Gini** — "how mixed is this group?" A tree splits on whatever makes the child groups purest.

**Precision** — of everyone we *called* a survivor, how many really were? (48/60 = 80%)

**Recall** — of all *real* survivors, how many did we catch? (48/69 = 70%)

**Accuracy** — how often right overall. Hides which kind of wrong; always read the confusion matrix too.

**Cross-validation** — k exams, different questions each, average the grades. Mean = best guess at true skill; spread = how much to trust that guess.

**Bias–variance tradeoff** — simple models are stable but wrong in a pattern (bias); flexible models fit perfectly but jump around with the data (variance). Good modeling sits in the middle. *Symptom reading: big train→val gap = too flexible; tiny gap + mediocre score = too rigid.*

**Stratified split** — split randomly *within each class* so the minority (survivors, 38%) keeps its share in every subset.

**Conditional probability** — P(survived | female) = 74%: restrict the world to a group, then measure. The seed idea behind evaluation, Bayes, everything.

**Drift detection (K-S / PSI)** — "do these two samples plausibly come from the same distribution?" K-S compares the curves; PSI measures how much bin shares moved. p < 0.05 or PSI > 0.25 → investigate.

**p-value** — the chance you'd see a difference this big by pure luck. Small p ⇒ act.
