"""CloudShield preprocessing — one module per dataset.

Each of the three tracks uses a different dataset with genuinely different
cleaning needs, so they do not share a pipeline (spec section 23). What they do
share is the methodology: fit every learned transform on the training split
only, keep the target out of X, and document every decision.

    from src.preprocessing.unsw_nb15 import get_classification_data
    from src.preprocessing.cse_cic_ids2018 import get_regression_data

Track 3 (TON_IoT clustering) is owned by Member 3 and its module is added when
that dataset is selected and documented.
"""

from . import cse_cic_ids2018, unsw_nb15

__all__ = ["unsw_nb15", "cse_cic_ids2018"]
