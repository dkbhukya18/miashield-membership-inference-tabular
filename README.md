# Preemptive Exclusion Against Membership Inference on Tabular Data (MIAShield ESE/ASE)

Course project for CIS 545 (Data Security and Privacy), University of Michigan-Dearborn, Fall 2023.
Team: Felipe Bastos, Ubong Imeh Effiom, Aloke Aggarwal, Dileep Kumar Bhukya. Advisor: Prof. Birhanu Eshete.

This repository is a **corrected and re-run** version of the original course submission. While preparing it for publication I found that the original evaluation could not measure membership leakage (details in [ERRATA.md](ERRATA.md)). The code here fixes those problems, and the numbers below come from the corrected pipeline, not the course report.

## What the project does

A membership inference attack (MIA) tries to tell whether a specific record was in a model's training set. MIAShield ([Eshete & Vorobeychik, 2022](https://arxiv.org/abs/2203.00915)) defends against this by **preemptive exclusion**:

1. Split the training data into *n* disjoint shards and train one model per shard.
2. At prediction time, an **exclusion oracle** checks whether the query is (or resembles) a training record. If so, the model trained on that record's shard is left out of the ensemble vote, so the answer comes only from models that never saw the record.

The original paper evaluated image data. This project applies two of its oracles to the UCI Bank Marketing dataset (45,211 customers, mixed numeric and categorical features):

- **ESE (Exact-Signature-based Exclusion):** SHA-1 hash of each training record; exclude only on an exact match.
- **ASE (Approximate-Signature-based Exclusion):** MinHash-LSH signatures (active categorical levels plus bucketed numeric values) retrieve similar training records, and a distance check (identical categories, every numeric feature within 0.03 in scaled units) confirms the match before excluding.

## Setup

- 80/20 stratified train/test split, 6 disjoint shards, seed 42.
- Target models: class-balanced logistic regression (as in the original project) and a random forest (added, because the logistic model leaks almost nothing; see results).
- Attack set: 5,000 true members (training records) and 5,000 true non-members (test records).
- Attacks:
  - The ART rule-based attack, which predicts "member" whenever the model's prediction is correct.
  - A confidence attack, scored with ROC-AUC on the model's confidence in the true label.
  - 0.50 means no leakage for both.
- Perturbed queries: the attacker adds Gaussian noise (σ = 0.01 on scaled numeric features), so the query is no longer byte-identical to the training record.

## Results (random forest target)

| Configuration | Queries | Test accuracy | Rule attack acc. | Confidence attack AUC | Members excluded |
|---|---|---|---|---|---|
| Single model, no defense | exact | 0.906 | 0.546 | **0.603** | n/a |
| Sharded ensemble, no oracle | exact | 0.903 | 0.510 | 0.518 | n/a |
| ESE | exact | 0.903 | 0.498 | **0.501** | 100% |
| ESE | perturbed | 0.903 | 0.505 | 0.509 | **0%** |
| ASE | perturbed | 0.903 | 0.499 | **0.504** | 98.2% (94.5% correct shard) |

Full results for both models are in [`results.csv`](results.csv).

### Takeaways

- **The logistic regression model does not leak membership on this data.** Attack accuracy and AUC stay at about 0.50 with or without a defense, because a regularized linear model barely overfits. The original report's "74% → 48%" result was an evaluation artifact, not a defense effect ([ERRATA.md](ERRATA.md), item 1).
- **On an overfitting model, leakage is real but modest**, with confidence-attack AUC of 0.60. Sharding alone removes most of it (0.52), and exact-match exclusion removes the rest (0.50). Test accuracy drops by only 0.3 points.
- **ESE breaks under trivial perturbation.** Once the query differs slightly from the stored record, the exact hash never matches (0% of members excluded), and leakage returns to the level of the unprotected ensemble.
- **ASE closes that gap**, matching 98% of perturbed member queries. The cost is falsely matching 20% of non-member queries to near-duplicate training records. That costs almost no accuracy here, but it shows how the approximation threshold trades protection against utility.

## Limitations

- Results come from one dataset, one split, and one seed, with no confidence intervals.
- The attacks are basic black-box attacks. Stronger ones, such as shadow-model or label-only attacks, were not evaluated.
- The random forest target was added after the course. It is not part of the original submission.
- The ASE thresholds (20 bins, 0.03 max distance) were tuned on this data. An adaptive attacker who knows them could perturb queries just beyond the threshold.

## How to run

```bash
pip install -r requirements.txt
python miashield_bank.py --data bank-full.csv   # about 2 minutes on CPU; writes results.csv and run_config.json
python audit_original.py                        # reproduces the flaws in the original evaluation
```

Dataset: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) (`bank-full.csv`, semicolon-separated).

## References

1. B. Eshete and Y. Vorobeychik. *MIAShield: Defending Membership Inference Attacks via Preemptive Exclusion of Members.* arXiv:2203.00915, 2022.
2. R. Shokri, M. Stronati, C. Song, V. Shmatikov. *Membership Inference Attacks Against Machine Learning Models.* IEEE S&P, 2017.
3. S. Yeom, I. Giacomelli, M. Fredrikson, S. Jha. *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting.* IEEE CSF, 2018.
4. Z. Li and Y. Zhang. *Membership Leakage in Label-Only Exposures.* ACM CCS, 2021.
5. S. Moro, P. Cortez, P. Rita. *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, 2014.
6. IBM Adversarial Robustness Toolbox (ART): https://github.com/Trusted-AI/adversarial-robustness-toolbox
7. datasketch (MinHash / LSH): https://github.com/ekzhu/datasketch
