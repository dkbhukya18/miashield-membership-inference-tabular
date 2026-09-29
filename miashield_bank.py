"""
MIAShield-style preemptive exclusion (ESE and ASE oracles) on the UCI Bank
Marketing dataset -- corrected and reproducible version of the CIS 545 project.

What changed vs. the original course notebooks (see ERRATA.md for details):
  * Ensemble members are genuinely trained on DISJOINT subsets (the original
    VotingClassifier.fit() silently retrained every member on the full data).
  * Membership ground truth is real: members come from the training set,
    non-members from the held-out test set, in equal numbers.
  * ASE hashes "column=value" tokens so categorical features actually count.
  * Every random step is seeded; results are regenerated, not hard-coded.
  * A high-capacity (overfitting) target model is added, because a regularized
    logistic regression barely leaks membership on this data.

Usage:  python miashield_bank.py --data bank-full.csv
"""
import argparse, hashlib, json, time, warnings
import numpy as np, pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
SEED = 42
N_MODELS = 6            # number of disjoint shards / ensemble members
N_EVAL = 5000           # members and non-members sampled for the attack
NOISE_SIGMA = 0.01      # attacker perturbation on scaled continuous features
N_BINS = 20             # ASE: continuous features bucketed into 20 bins for the signature
LSH_THRESHOLD = 0.5     # ASE: Jaccard threshold for LSH candidate retrieval
MAX_CONT_DIST = 0.03    # ASE: max per-feature difference (scaled units) to accept a match (~3x noise sigma)


# ----------------------------------------------------------------- data
def load(path):
    df = pd.read_csv(path, delimiter=";")
    cats = ['job', 'marital', 'education', 'default', 'housing', 'loan',
            'contact', 'month', 'poutcome']
    y = (df.pop('y') == 'yes').astype(int)
    X = pd.get_dummies(df, columns=cats, drop_first=True).astype(float)
    cont = [c for c in ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']]
    X[cont] = (X[cont] - X[cont].min()) / (X[cont].max() - X[cont].min())
    return X, y, cont


# ----------------------------------------------------------------- models
def make_model(kind):
    if kind == "logreg":
        return LogisticRegression(class_weight='balanced', solver='newton-cg',
                                  max_iter=1000, random_state=SEED)
    return RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)


class ShardedEnsemble:
    """N models, each trained on one disjoint shard of the training data."""
    def __init__(self, kind, X_train, y_train, n=N_MODELS):
        rng = np.random.RandomState(SEED)
        idx = rng.permutation(X_train.index)
        self.shards = np.array_split(idx, n)
        self.models = [make_model(kind).fit(X_train.loc[s], y_train.loc[s]) for s in self.shards]
        self.shard_of = {i: k for k, s in enumerate(self.shards) for i in s}

    def proba_matrix(self, X):
        return np.stack([m.predict_proba(X)[:, 1] for m in self.models])  # (n, N)

    def predict_proba(self, X, excluded=None):
        """excluded[j] = shard index to leave out for query j (-1 = none)."""
        P = self.proba_matrix(X)
        if excluded is None:
            return P.mean(axis=0)
        excluded = np.asarray(excluded)
        out = P.mean(axis=0)
        hit = excluded >= 0
        cols = np.where(hit)[0]
        out[hit] = (P[:, hit].sum(axis=0) - P[excluded[hit], cols]) / (P.shape[0] - 1)
        return out


# ----------------------------------------------------------------- oracles
def row_sha1(row):
    return hashlib.sha1(np.round(row, 10).tobytes()).hexdigest()


class ExactOracle:
    """ESE: exclude a shard only if the query is byte-identical to a training record."""
    def __init__(self, X_train, ens):
        self.lookup = {row_sha1(r): ens.shard_of[i]
                       for i, r in zip(X_train.index, X_train.to_numpy())}

    def __call__(self, X):
        return np.array([self.lookup.get(row_sha1(r), -1) for r in X.to_numpy()])


class ApproxOracle:
    """ASE: MinHash-LSH finds candidate training records with a similar signature,
    then a distance check confirms the match before a shard is excluded.

    Signature tokens = the active categorical levels + bucketed continuous values.
    (Hashing every one-hot column, including all the zeros, makes almost every pair
    of rows look ~90% similar, which is why a verification step is required.)"""
    def __init__(self, X_train, ens, cont_cols):
        self.cols = np.array(X_train.columns); self.ens = ens
        self.cont_mask = np.isin(self.cols, cont_cols)
        self.train = X_train
        self.lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=128)
        sigs = self._minhashes(X_train.to_numpy())
        with self.lsh.insertion_session() as session:
            for i, m in zip(X_train.index, sigs):
                session.insert(int(i), m)
        self._cache = {}

    def _tokens(self, r):
        cont = [f"{c}={int(v * N_BINS)}".encode() for c, v in zip(self.cols[self.cont_mask], r[self.cont_mask])]
        cat = [f"{c}".encode() for c, v in zip(self.cols[~self.cont_mask], r[~self.cont_mask]) if v >= 0.5]
        return cont + cat

    def _minhashes(self, A):
        return MinHash.bulk([self._tokens(r) for r in A], num_perm=128, seed=SEED)

    def __call__(self, X):
        key = (id(X), len(X))
        if key in self._cache:
            return self._cache[key]
        A = X.to_numpy(); out = []
        for r, q in zip(A, self._minhashes(A)):
            cands = self.lsh.query(q)
            if not cands:
                out.append(-1); continue
            C = self.train.loc[cands].to_numpy()
            same_cat = (np.round(C[:, ~self.cont_mask]) == np.round(r[~self.cont_mask])).all(axis=1)
            dist = np.abs(C[:, self.cont_mask] - r[self.cont_mask]).max(axis=1)
            ok = same_cat & (dist <= MAX_CONT_DIST)
            if not ok.any():
                out.append(-1); continue
            best = np.array(cands)[ok][np.argmin(dist[ok])]
            out.append(self.ens.shard_of[int(best)])
        self._cache[key] = np.array(out)
        return self._cache[key]


# ----------------------------------------------------------------- attacks
def evaluate_attack(p1, y):
    """p1 = P(y=1) returned to the attacker for each query; y = true labels.
    First half of the queries are members, second half non-members."""
    n = len(y) // 2
    member = np.r_[np.ones(n), np.zeros(n)]
    correct = ((p1 >= 0.5).astype(int) == y)
    rule_acc = (correct == member).mean()                 # ART rule-based attack logic
    conf_true = np.where(y == 1, p1, 1 - p1)              # confidence on the true label
    auc = roc_auc_score(member, conf_true)                # threshold-free confidence attack
    return rule_acc, auc


def main(path):
    t0 = time.time()
    X, y, cont = load(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                        random_state=SEED, stratify=y)
    rng = np.random.RandomState(SEED)
    mem = rng.choice(X_train.index, N_EVAL, replace=False)
    non = rng.choice(X_test.index, N_EVAL, replace=False)
    Q = pd.concat([X.loc[mem], X.loc[non]]); yq = np.r_[y.loc[mem], y.loc[non]]
    Qp = Q.copy()                                          # attacker-perturbed queries
    Qp[cont] = np.clip(Qp[cont] + rng.normal(0, NOISE_SIGMA, Qp[cont].shape), 0, 1)

    results = []
    oracles = None
    for kind in ["logreg", "random_forest"]:
        print(f"[{time.time()-t0:6.1f}s] training {kind}", flush=True)
        single = make_model(kind).fit(X_train, y_train)
        ens = ShardedEnsemble(kind, X_train, y_train)
        if oracles is None:   # shard assignment only depends on SEED, so build the oracles once
            print(f"[{time.time()-t0:6.1f}s] building ESE/ASE oracles", flush=True)
            oracles = ExactOracle(X_train, ens), ApproxOracle(X_train, ens, cont)
        ese, ase = oracles

        def row(defense, query_type, p_query, p_test, excl=None):
            ra, auc = evaluate_attack(p_query, yq)
            r = dict(model=kind, defense=defense, queries=query_type,
                     test_accuracy=round(float(((p_test >= 0.5) == y_test).mean()), 4),
                     rule_attack_accuracy=round(float(ra), 4), confidence_attack_auc=round(float(auc), 4))
            if excl is not None:
                truth = np.array([ens.shard_of[i] for i in mem])
                r["member_exclusion_rate"] = round(float((excl[:N_EVAL] >= 0).mean()), 4)
                r["member_correct_shard_rate"] = round(float((excl[:N_EVAL] == truth).mean()), 4)
                r["nonmember_false_exclusion_rate"] = round(float((excl[N_EVAL:] >= 0).mean()), 4)
            results.append(r)

        p_test_full = ens.predict_proba(X_test)
        row("none (single model)", "exact", single.predict_proba(Q)[:, 1], single.predict_proba(X_test)[:, 1])
        row("none (sharded ensemble)", "exact", ens.predict_proba(Q), p_test_full)
        for name, oracle in [("ESE", ese), ("ASE", ase)]:
            p_test = ens.predict_proba(X_test, oracle(X_test))
            for qt, QQ in [("exact", Q), ("perturbed", Qp)]:
                ex = oracle(QQ)
                row(name, qt, ens.predict_proba(QQ, ex), p_test, ex)
        row("none (sharded ensemble)", "perturbed", ens.predict_proba(Qp), p_test_full)

    res = pd.DataFrame(results)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print(res.to_string(index=False))
    res.to_csv("results.csv", index=False)
    json.dump(dict(seed=SEED, n_models=N_MODELS, n_eval_each=N_EVAL, noise_sigma=NOISE_SIGMA,
                   ase_bins=N_BINS, lsh_threshold=LSH_THRESHOLD, max_cont_dist=MAX_CONT_DIST, runtime_s=round(time.time() - t0, 1)),
              open("run_config.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="bank-full.csv")
    main(ap.parse_args().data)
