# Model card: credit risk adjudication model

Version 0.1.0. All figures are from `python -m creditadj all` at seed 42 and are stored in `reports/metrics.json`.

## Purpose and intended use

The model estimates the probability that a credit card account defaults on its next payment, and turns that estimate into an approve or decline decision with stated reasons.

It exists to show a complete adjudication workflow on public data: feature design, a constrained model, a benchmark, a cutoff with its costs, reason codes, a fair-lending outcome check and a monitoring plan.

**Intended use.** Teaching, review and discussion. Scoring the preset applicants in the web demo.

**Not intended for.** Any real credit decision. It was built on 2005 data from one bank in Taiwan and has never been tested on later accounts, another market or real applicants.

## Data and its limits

| | |
|---|---|
| Source | UCI Machine Learning Repository, "Default of Credit Card Clients", https://doi.org/10.24432/C55S3H, CC BY 4.0 |
| Paper | Yeh and Lien (2009), Expert Systems with Applications 36, 2473-2480 |
| Population | 30,000 existing credit card holders at one bank in Taiwan |
| Observation window | April to September 2005: monthly repayment status, statement balance, payment amount |
| Outcome | "Default payment next month", 1 or 0. 22.1% of accounts are 1 |
| Split | Stratified random 60 / 20 / 20: 18,000 train, 6,000 validation, 6,000 test |

Limits that matter for anyone reading the results:

- **One snapshot.** All accounts share the same six months and the same outcome month. There are no dates, so there is no out-of-time test and no way to measure how fast the model decays.
- **Existing customers.** The inputs are behaviour on the card. Real applicants do not have that history with the lender, and a real application model has to handle the fact that outcomes are only observed for approved accounts. Neither issue arises here, which makes the problem easier than the real one.
- **A thin outcome.** The source does not define the default flag in days past due, and the window is one month. The 22.1% rate is far higher than a normal card book. The source paper describes a card-debt crisis in Taiwan at the time.
- **Undocumented codes.** Repayment status values -2 and 0 are not in the UCI documentation. My reading of them, and the evidence for it, is in the README.
- **Dormant accounts.** 966 accounts have no balance on any statement and 35.9% of them are flagged as defaults. I cannot explain that from the data.

## Features

Twelve engineered features, each with a monotonic constraint. Definitions are in `creditadj/features.py` and in the README.

| Feature | Direction of risk |
|---|---|
| `credit_limit` | lower limit, higher risk |
| `recent_status` | more months past due now, higher risk |
| `worst_status_6m` | deeper delinquency in six months, higher risk |
| `months_late_6m` | more late months, higher risk |
| `months_since_late` | more recent late payment, higher risk |
| `util_current` | higher utilisation, higher risk |
| `util_avg_6m` | higher average utilisation, higher risk |
| `util_trend` | rising utilisation, higher risk |
| `pay_ratio` | smaller share of bills repaid, higher risk |
| `last_payment` | smaller last payment, higher risk |
| `months_paid_full` | fewer statements paid in full, higher risk |
| `months_zero_balance` | more statements with no balance, higher risk |

The last direction is taken from this dataset, not from prior belief. The others are standard.

## Excluded attributes and why

| Attribute | Reason |
|---|---|
| SEX | Prohibited basis under the US Equal Credit Opportunity Act and Regulation B. Prohibited ground under the Canadian Human Rights Act |
| MARRIAGE | Same |
| AGE | Same. Regulation B permits narrow uses of age in a validated scoring system. I did not rely on that |
| EDUCATION | Not prohibited in itself, but a possible proxy for protected characteristics, and not needed when repayment behaviour is observed directly |

Cost of excluding them: the same model with all four added scores 0.771 AUC on the test set against 0.768, a gain of 0.002 (paired bootstrap interval -0.000 to 0.005).

## Model

- **Champion.** XGBoost binary classifier, 200 trees, depth 3, learning rate 0.05, histogram method, monotonic constraints on all 12 features, starting score fixed at the training default rate, no class weights.
- **Benchmark.** Logistic regression on the same features, currency amounts logged, all inputs standardised, L2 penalty with C = 1.
- **Calibration.** Platt scaling was fitted on validation as a check (slope 0.975, intercept -0.011). The rule set beforehand was to apply it only if either differed from no change by more than 0.10. It is not applied.
- **No tuning.** Hyperparameters were fixed in advance. Feature and constraint choices were made on the validation set.

## Performance

Test set, 6,000 accounts.

| | AUC | Gini | KS |
|---|---|---|---|
| Champion | 0.768 | 0.537 | 0.418 |
| Benchmark | 0.753 | 0.505 | 0.398 |

- Champion AUC, 95% bootstrap interval: 0.753 to 0.784.
- Champion minus benchmark: 0.016 (0.010 to 0.021).
- Train AUC 0.787, validation 0.776, test 0.768. Train minus test: 0.019.
- Without monotonic constraints the test AUC is 0.779. The constraints cost 0.011.
- Default rate by score decile runs from 5.3% to 69.0%. One inversion: decile 7 (17.0%) is below decile 6 (17.7%).
- Calibration: mean predicted 22.05%, observed 22.12%. Brier score 0.1372. Expected calibration error over deciles 0.013. Largest decile gap 2.9 points, in decile 5.

## Fairness check

Approval rates on the test set at the recommended cutoff of 0.13. Adverse-impact ratio is the lowest group approval rate divided by the highest.

| Attribute | Groups (approval rate) | Adverse-impact ratio |
|---|---|---|
| Sex | Male 42.9%, Female 45.0% | 0.954 |
| Age band | 21-29 38.9%, 30-39 49.5%, 40-49 44.7%, 50+ 39.7% | 0.786 |
| Marital status | Married 45.5%, Single 43.3%, Other 32.0% (75 accounts) | 0.704 |

The age ratio is below the 0.80 screen. Giving every account the same credit limit raises it to 0.936, so the limit is the main route by which age reaches the decision (median limit NT$90,000 for ages 21-29, NT$180,000 for 30-39). The marital-status ratio rests on a group of 75 accounts.

The model has not been changed in response to these results. They are reported as found.

Not checked: race, ethnicity or national origin (not in the data), intersections of groups, other cutoffs, error-rate parity beyond the approval rate among non-defaulters shown in the README.

## Cutoff recommendation and assumptions

Approve when the predicted probability of default is below **0.13**.

| On the test set | |
|---|---|
| Approval rate | 44.1% |
| Default rate among approved | 9.0% |
| Defaulters declined | 82.0% |
| Defaulted balance declined | 70.4% |
| Good accounts declined | 48.4% |

The cutoff maximises an illustrative profit measure on the validation set. That measure rests on two assumptions that are not estimated from data:

- loss given default of 75% of the most recent statement balance
- annual margin of 12% of the balance on accounts that do not default

Break-even probability of default under these is 13.8%. With a 75% loss and margins from 8% to 16% the chosen cutoff is 0.12 or 0.13. At a 4% margin it is 0.04, and at 20% it is 0.19. On the test set itself the best cutoff is 0.12, and 0.13 reaches 87% of its profit.

The measure ignores recoveries, cure rates, fee and interchange income, funding cost and anything beyond the first month.

## Reason-code method

For a declined applicant, each feature in turn is replaced by a reference value (the median among approved, non-defaulting training accounts) and the applicant is re-scored. Features are ranked by the resulting drop in probability. The top four with a drop of at least 0.001 are reported using fixed sentences in `creditadj/reasons.py`.

- Every feature is monotone, so each sentence states a direction that is guaranteed to match the applicant's value. This is tested.
- It is a one-at-a-time perturbation, not a Shapley decomposition. Drops do not sum to the total gap, and the result depends on the reference point.
- The delinquency features overlap, so several of the four reasons can describe the same event.
- All 3,352 declined test accounts receive at least one reason. 896 receive fewer than four.
- The most common first reason is a low most-recent payment (28.0% of declines), then a recent late payment (18.4%).
- Against XGBoost's `pred_contribs` on 500 declined test accounts: same top reason in 86.4% of cases, top reason within its top four in 100.0%, average top-four overlap 88.9%.

## Monitoring plan

Population stability index thresholds: below 0.10 stable, 0.10 to 0.25 investigate, 0.25 and above review.

| Frequency | What to track |
|---|---|
| Monthly | PSI of the score and of each feature against the development sample. Approval rate, override rate, mix of reason codes. New or unexpected values in the status fields. Early delinquency on recent approvals |
| Quarterly | AUC, Gini and KS on matured accounts. Observed against predicted default rate by decile, especially the deciles around the cutoff. Adverse-impact ratios. Cutoff table re-run with actual margin and loss figures |
| Annually | Full revalidation. Challenger refit on recent data. Each monotonic constraint re-tested. Reason-code reference values refreshed. Assumptions and this card reviewed |

Proposed triggers for an early review: score PSI at or above 0.25, AUC more than 0.03 below development, or an adverse-impact ratio under 0.80 in two consecutive quarters. These thresholds are my proposals and have not been tested against real monitoring data.

In the demonstration in the README, a stressed copy of the test set pushes `util_current` to a PSI of 0.321 while the score PSI stays at 0.040. Input drift can be large before the score shows it.

## Known weaknesses

1. No out-of-time validation is possible.
2. The population is existing cardholders, not applicants.
3. The outcome definition is vague and the window is one month.
4. Calibration is weakest in deciles 5 and 6, at and just above the cutoff: predicted 13.1% and 15.0%, observed 16.0% and 17.7%.
5. The age adverse-impact ratio is below 0.80, mainly through credit limit.
6. `months_zero_balance` and `last_payment` let the model penalise inactive accounts. The pattern is strong in this data and unexplained.
7. `credit_limit` is an earlier decision by the same lender. Using it feeds past judgement back into new decisions.
8. The monotonic constraints cost 0.011 AUC on the test set.
9. The recommended cutoff declines 48.4% of customers who would have paid, and it rests on assumed economics.
10. Results are for one seed and one split. The bootstrap interval does not cover variation between splits.

## What I would do next with real lender data

- Build on application and bureau data at the point of decision, with an outcome such as 90 days past due within 12 or 24 months.
- Split by time: develop on older vintages, hold out the most recent ones, and report performance by vintage.
- Address reject inference, since outcomes would be missing for declined applicants.
- Replace the two assumed numbers with the lender's own loss, recovery and margin figures, and choose the cutoff on expected profit per account over the life of the loan.
- Run a search for less discriminatory alternatives: for each candidate model, report both predictive performance and adverse-impact ratios, starting with the credit-limit finding.
- Have compliance review the reason-code sentences, and test them with customers.
- Repeat the split over many seeds to see how much the results move.
- Set up the monitoring above on live data and agree the triggers with the model owner before launch.
