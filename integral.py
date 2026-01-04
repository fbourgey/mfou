import numpy as np
from scipy import integrate, special


def I_closed_form_H_2(h, a_i, a_j):
    """
    Compute closed-form integral for the case H_i + H_j = 2.

    Parameters
    ----------
    h : float or ndarray
        Upper integration limit.
    a_i, a_j : float
        Parameters of the exponential terms.

    Returns
    -------
    float or ndarray
        Value of the integral.
    """
    return (np.exp(a_i * h) - 1.0) / (a_i * a_j)


def I_closed_form(h, a_i, a_j, H_i, H_j, opt="closed_form"):
    """
    Compute integral using closed-form or single integral representation.

    Parameters
    ----------
    h : float
        Upper integration limit.
    a_i, a_j : float
        Exponential parameters.
    H_i, H_j : float
        Hurst parameters, must satisfy 0 < H_i + H_j <= 2.
    opt : {'closed_form', 'single_integral'}, default='closed_form'
        Method for computation.

    Returns
    -------
    float
        Value of the integral.

    Raises
    ------
    ValueError
        If H_i + H_j not in (0, 2] or invalid opt.
    """
    H_ij = H_i + H_j
    if H_ij < 0 or H_ij > 2:
        raise ValueError("H_i + H_j must be in (0, 2]")
    if H_ij == 2:
        return I_closed_form_H_2(h, a_i, a_j)
    else:
        if opt == "single_integral":
            # single integral form
            def integrand(u):
                return (
                    a_j ** (1 - H_ij)
                    * np.exp((a_i + a_j) * u)
                    * (
                        special.gammaincc(H_ij, a_j * u) * special.gamma(H_ij)
                        - (a_j * u) ** (H_ij - 1) * np.exp(-a_j * u)
                    )
                    / (H_ij - 1)
                )

            return integrate.quad(integrand, 0, h)[0]

        elif opt == "closed_form":
            # closed form
            res = (
                a_j ** (1 - H_ij)
                * np.exp((a_i + a_j) * h)
                * special.gammaincc(H_ij, a_j * h)
                * special.gamma(H_ij)
                - h ** (H_ij - 1) * np.exp(a_i * h)
                - a_j ** (1 - H_ij) * special.gamma(H_ij)
                + h ** (H_ij - 1) * special.hyp1f1(H_ij - 1, H_ij, a_i * h)
            )
            return res / ((a_i + a_j) * (H_ij - 1))
        else:
            raise ValueError("opt must be in {'single_integral', 'closed_form'}")


def I_quad(h, a_i, a_j, H_i, H_j):
    """
    Compute integral using numerical double integration.

    Parameters
    ----------
    h : float
        Upper integration limit.
    a_i, a_j : float
        Exponential parameters.
    H_i, H_j : float
        Hurst parameters.

    Returns
    -------
    float
        Numerically integrated value.
    """

    def inner_int_quad(u):
        result, _ = integrate.quad(
            lambda v: np.exp(a_i * u + a_j * v) * (u - v) ** (H_i + H_j - 2), -np.inf, 0
        )
        return result

    return integrate.quad(inner_int_quad, 0, h)[0]
