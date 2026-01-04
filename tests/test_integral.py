import pytest
from integral import I_closed_form, I_quad, I_closed_form_H_2
import numpy as np


@pytest.mark.parametrize(
    "a_i, a_j, H_i",
    list(
        zip(
            np.linspace(1e-3, 3.0, 20),
            np.linspace(1e-3, 3.0, 20),
            np.linspace(0.0, 2.0, 20),
            strict=False,
        )
    ),
)
def test_integral_H_i_H_j_equal_2(a_i, a_j, H_i):
    H_j = 2.0 - H_i
    hs = np.linspace(1e-3, 3.0, 100)

    double_integral = np.array([I_quad(h, a_i, a_j, H_i, H_j) for h in hs])
    closed_form = I_closed_form_H_2(hs, a_i, a_j)
    single_integral = np.array(
        [I_closed_form(h, a_i, a_j, H_i, H_j, opt="single_integral") for h in hs]
    )

    assert np.allclose(double_integral, closed_form, rtol=1e-10)
    assert np.allclose(double_integral, single_integral, rtol=1e-10)
