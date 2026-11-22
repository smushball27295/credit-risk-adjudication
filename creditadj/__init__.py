"""Credit risk adjudication model on the UCI "Default of Credit Card Clients" data.

The package is organised by the questions a lender asks:

    data.py      where the data comes from and how it is checked
    features.py  the 12 engineered inputs the model is allowed to see
    model.py     champion (XGBoost) and benchmark (logistic regression)
    evaluate.py  discrimination, calibration, stability of the estimate
    cutoff.py    where to put the approval cutoff and what it costs
    reasons.py   why an applicant was declined
    fairness.py  approval rates by sex and age band
    monitor.py   population stability index
    export.py    one JSON file a browser can score from
"""

__version__ = "0.1.0"

SEED = 42
