import pytest


def test_bsqp_reports_unsupported_horizon_through_public_constructor():
    from bsqp.interface import BSQP

    with pytest.raises(ValueError, match="Number of knots 999 not supported"):
        BSQP(
            model_path="examples/indy7_description/indy7.urdf",
            batch_size=1,
            N=999,
            dt=0.01,
            plant_type="indy7",
        )
