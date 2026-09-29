"""Reproduce the original CIS 545 evaluation logic and compare it to a correct
membership-inference evaluation. Used only to verify the discrepancies."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from art.estimators.classification import SklearnClassifier
from art.attacks.inference.membership_inference import MembershipInferenceBlackBoxRuleBased

import sys
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "bank-full.csv", delimiter=";")
cats = ['job','marital','education','default','housing','loan','contact','month','poutcome','y']
df[cats] = df[cats].astype('category')
df = pd.concat((df, pd.get_dummies(df[cats], drop_first=True)), axis=1)
X = df.drop(columns=cats + ['y_yes']).astype(float)
y = df['y_yes'].astype(int)
cont = X.iloc[:, :7]; disc = X.iloc[:, 7:]
cont = (cont - cont.min()) / (cont.max() - cont.min())
X = pd.concat((cont, disc), axis=1); X.insert(0, 'const', 1.0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
lr = lambda: LogisticRegression(class_weight='balanced', solver='newton-cg', max_iter=1000, random_state=42)
base = lr().fit(X_train, y_train)
print("Base model test accuracy (sklearn predict, 0.5 threshold):", round(base.score(X_test, y_test), 4))

def rule_attack(model, Xm, ym, Xn, yn):
    atk = MembershipInferenceBlackBoxRuleBased(SklearnClassifier(model=model))
    im = atk.infer(Xm.to_numpy(), ym.to_numpy()); inn = atk.infer(Xn.to_numpy(), yn.to_numpy())
    tr = im.mean(); te = 1 - inn.mean()
    return (tr*len(im) + te*len(inn)) / (len(im) + len(inn))

# ---- ORIGINAL evaluation: attacker's "members" = random 85% of the whole dataset,
# "non-members" = random 15%. These labels have nothing to do with real membership.
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(X, y, test_size=0.15, random_state=23)
print("ORIGINAL-style baseline attack accuracy:", round(rule_attack(base, Xa_tr, ya_tr, Xa_te, ya_te), 4))
acc_all = base.score(X, y)
print("  predicted by artifact formula 0.85*acc + 0.15*(1-acc):", round(0.85*acc_all + 0.15*(1-acc_all), 4))

# Original 'ensemble': VotingClassifier.fit() re-trains every member on ALL of X_train
sub = [X_train.iloc[i::6] for i in range(6)]
vc = VotingClassifier([(str(i), lr()) for i in range(6)], voting='soft').fit(X_train, y_train)
same = all(np.allclose(e.coef_, vc.estimators_[0].coef_) for e in vc.estimators_)
print("All 6 'disjoint' ensemble members have identical weights after VotingClassifier.fit:", same)

# ORIGINAL 'protected' evaluation: both member and non-member sets are ~the whole dataset
Xm = X.drop(Xa_tr.index[:500]); Xn = X.drop(Xa_te.index[:100])
print("ORIGINAL-style 'protected' attack accuracy (near-identical sets):",
      round(rule_attack(base, Xm, y.loc[Xm.index], Xn, y.loc[Xn.index]), 4))

# ---- CORRECT evaluation: true members vs true non-members, balanced
rng = np.random.RandomState(0)
mi = rng.choice(X_train.index, 5000, replace=False); ni = rng.choice(X_test.index, 5000, replace=False)
print("CORRECT baseline attack accuracy (true members vs non-members):",
      round(rule_attack(base, X.loc[mi], y.loc[mi], X.loc[ni], y.loc[ni]), 4))
