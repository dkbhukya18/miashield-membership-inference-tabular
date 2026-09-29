# Errata: original CIS 545 report and notebooks

These issues were found by auditing `ESE_Final_Project.ipynb`, `ase_oracle_group_2_cis_545.py`, and the submitted report (Dec 2023). Items 1–3 were confirmed by re-running the original logic (`audit_original.py`).

## Evaluation and code

**1. The baseline attack accuracy (0.74) did not measure membership.**
The attacker's "members" and "non-members" were a random 85/15 split of the *entire* dataset (`train_test_split(X_scaled, y, test_size=0.15, random_state=23)`), unrelated to which records the model was trained on. Under the rule-based attack, accuracy then reduces to 0.85·acc + 0.15·(1 − acc), which is a function of model accuracy alone.
- Re-running the original logic gives 0.7427. The formula predicts 0.742.
- With true members vs. non-members, the same model scores **0.506**, meaning essentially no leakage.

**2. The "protected" attack accuracy (about 0.48) was also an artifact.**
The member and non-member evaluation sets were both "the whole dataset minus a few rows" (`pd.concat([X_scaled, X_attack_train_[i]]).drop_duplicates(keep=False)`). They were nearly identical, so train_acc + test_acc ≈ 1 and the weighted accuracy is about 0.5 by construction.

**3. The ensemble was not trained on disjoint subsets.**
`VotingClassifier.fit(X, y)` clones and **re-trains every member on the full data passed to `fit`**. After fitting, all six "disjoint" models had identical weights (verified). This was also why the "ensemble" attack accuracy matched the single model exactly (0.7427 both). The core premise of MIAShield, that each record influences only one model, was never implemented.

**4. The "accuracies of disjoint models" figure (report Image 3) shows the wrong model.**
`prediction(X_sample_[i], coeffs)` used the **base model's** coefficients, not `coeffs_[i]`. The six numbers are the base model evaluated on each shard. Shard sampling also had no seed, so the notebook's own outputs (0.8736, 0.8769, …) differ from the report's (0.8663, 0.8704, …).

**5. The base model accuracy (0.869) used a non-default decision rule.**
The custom `prediction()` ignores the model's intercept, and `evaluation()` uses a 0.75 threshold even though its comment says 0.5. The standard `model.predict` accuracy is 0.844.

**6. The ASE point on the results plot was hard-coded.**
`plt.scatter(p_model_acc, 0.477, label='ASE Protected')` hard-codes 0.477 rather than computing it. The x-coordinate for both protected points, `p_model_acc`, is the base model's accuracy on the last shard (see item 4), not the protected ensemble's test accuracy.

**7. The ASE MinHash ignored column identity.**
Each row was hashed as a *set of values* (`minhash.update(str(value))`), so all 35+ one-hot columns collapsed into the two tokens "0" and "1". The signature therefore reflected only the unperturbed numeric columns, which is why the report observed that the first neighbor was always at distance 0. ASE was effectively exact matching.

**8. Other code issues.**
- The fallback model for "no match" queries (`ensemble_[n+1]`) was re-trained on `X_test`, so non-member queries were answered by a model trained on them.
- The perturbation loop `if j < 9` skips index 8, which is the first one-hot column.
- `!pip hash` in the setup does nothing.
- `DataFrame.append` was removed in pandas 2.0.

## Report text

**9. References [2]–[5] are not real publications.**
Items [2]–[4] (e.g., "Smith & Jones, *Advancements in Cybersecurity: A Comprehensive Overview*") could not be found, and [5] is leftover filler text. Replaced with verifiable sources in the README.

**10. ESE was described incorrectly.**
The report describes ESE as signatures for "malicious code" and "known threats". In MIAShield, ESE hashes **training records** so the system can detect when a query is a training member. It has nothing to do with malware signatures.

**11. An irrelevant citation was used for ASE.**
The ASE background cites a road-network spatial keyword query paper [6] that does not relate to MIAShield.

**12. The framing was misstated.**
The abstract describes the goal as "safeguarding marketing strategies". The actual goal is protecting the privacy of the customers whose records trained the model.

## What changed in this repository

`miashield_bank.py` makes the following changes:
- Trains each member only on its own shard, averaging predictions manually instead of refitting through `VotingClassifier`.
- Evaluates attacks on true members vs. true non-members.
- Hashes column-aware tokens and verifies each ASE candidate by distance.
- Seeds every random step.
- Reports test accuracy, attack accuracy, attack AUC, and oracle exclusion rates.
- Adds perturbed-query and random-forest experiments.
