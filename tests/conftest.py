import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pull-tests")

import pytest

from model_fitting import PoissonPAFit, TailFit, TimeAggregation


@pytest.fixture
def fit():
    return PoissonPAFit(
        lambda_hat=2.0,
        p_hat=0.5,
        delta_in_hat=1.0,
        delta_out_hat=1.0,
        iota_in_hat=2.0,
        iota_out_hat=2.0,
        edge_count=40,
        node_count=22,
        new_node_count=20,
        hour_mode="active",
        time_aggregation=TimeAggregation(3600, 0, 19 * 3600, 0, 19, 20, 20, 40),
        in_tail=TailFit(2.0, 10, 1.0, 0.1, 20),
        out_tail=TailFit(2.0, 10, 1.0, 0.1, 20),
    )
