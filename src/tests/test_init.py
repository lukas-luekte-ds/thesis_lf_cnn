import math

import torch

from complex_methods import initialize_complex_weights


# Large matrices so that empirical moments are close
# to their theoretical values.
FAN_IN = 1000
FAN_OUT = 500

REL_TOL = 0.03
ABS_MEAN_TOL = 0.005


def get_expected_variances(
    method: str,
    fan_in: int,
    fan_out: int,
    negative_slope: float | None = None,
) -> tuple[float, float]:
    """
    Returns:
        expected_component_variance:
            Var(W_R) = Var(W_I)

        expected_complex_variance:
            E[|W|^2] = E[W_R^2 + W_I^2]
            for zero-mean weights.
    """

    if method == "componentwise_pytorch_default":
        component_var = 1.0 / (3.0 * fan_in)

    elif method in {
        "split_relu_he_normal",
        "split_relu_he_uniform",
    }:
        component_var = 1.0 / fan_in

    elif method in {
        "split_leakyrelu_he_normal",
        "split_leakyrelu_he_uniform",
    }:
        if negative_slope is None:
            raise ValueError(
                "negative_slope required for LeakyReLU initialization"
            )

        component_var = (
            1.0
            / (
                (1.0 + negative_slope**2)
                * fan_in
            )
        )

    elif method in {
        "split_relu_xavier_normal",
        "split_relu_xavier_uniform",
    }:
        component_var = (
            2.0
            / (fan_in + fan_out)
        )

    elif method in {
        "split_leakyrelu_xavier_normal",
        "split_leakyrelu_xavier_uniform",
    }:
        if negative_slope is None:
            raise ValueError(
                "negative_slope required for LeakyReLU initialization"
            )

        component_var = (
            2.0
            / (
                (1.0 + negative_slope**2)
                * (fan_in + fan_out)
            )
        )

    else:
        raise ValueError(
            f"No theoretical variance defined for method: {method}"
        )

    complex_var = 2.0 * component_var

    return component_var, complex_var


def test_initialization(initialization: dict) -> None:
    """
    Tests one complex initialization method.
    """

    method = initialization["method"]

    # Create empty real and imaginary weight matrices.
    weight_real = torch.empty(FAN_OUT, FAN_IN)
    weight_imag = torch.empty(FAN_OUT, FAN_IN)

    # Apply initialization.
    initialize_complex_weights(
        weight_real=weight_real,
        weight_imag=weight_imag,
        fan_in=FAN_IN,
        fan_out=FAN_OUT,
        initialization=initialization,
    )

    # ---------------------------------------------------------
    # Empirical statistics
    # ---------------------------------------------------------

    mean_real = weight_real.mean().item()
    mean_imag = weight_imag.mean().item()

    var_real = weight_real.var(unbiased=False).item()
    var_imag = weight_imag.var(unbiased=False).item()

    # For zero-mean complex weights:
    #
    # Var_C(W) = E[|W|^2]
    #          = E[W_R^2 + W_I^2]
    #
    complex_var = (
        weight_real.pow(2)
        + weight_imag.pow(2)
    ).mean().item()

    # ---------------------------------------------------------
    # Theoretical statistics
    # ---------------------------------------------------------

    expected_component_var, expected_complex_var = (
        get_expected_variances(
            method=method,
            fan_in=FAN_IN,
            fan_out=FAN_OUT,
            negative_slope=initialization.get("negative_slope"),
        )
    )

    # ---------------------------------------------------------
    # Print result
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print(f"Method: {method}")
    print("=" * 70)

    print(
        f"Mean(W_R): "
        f"expected ~ 0.0 | actual = {mean_real:.8f}"
    )

    print(
        f"Mean(W_I): "
        f"expected ~ 0.0 | actual = {mean_imag:.8f}"
    )

    print()

    print(
        f"Var(W_R): "
        f"expected = {expected_component_var:.8f} | "
        f"actual = {var_real:.8f}"
    )

    print(
        f"Var(W_I): "
        f"expected = {expected_component_var:.8f} | "
        f"actual = {var_imag:.8f}"
    )

    print(
        f"E[|W|²]:  "
        f"expected = {expected_complex_var:.8f} | "
        f"actual = {complex_var:.8f}"
    )

    # ---------------------------------------------------------
    # Assertions
    # ---------------------------------------------------------

    assert abs(mean_real) < ABS_MEAN_TOL, (
        f"{method}: Mean of W_R is too far from zero: "
        f"{mean_real}"
    )

    assert abs(mean_imag) < ABS_MEAN_TOL, (
        f"{method}: Mean of W_I is too far from zero: "
        f"{mean_imag}"
    )

    assert math.isclose(
        var_real,
        expected_component_var,
        rel_tol=REL_TOL,
    ), (
        f"{method}: Wrong variance for W_R. "
        f"Expected {expected_component_var}, got {var_real}"
    )

    assert math.isclose(
        var_imag,
        expected_component_var,
        rel_tol=REL_TOL,
    ), (
        f"{method}: Wrong variance for W_I. "
        f"Expected {expected_component_var}, got {var_imag}"
    )

    assert math.isclose(
        complex_var,
        expected_complex_var,
        rel_tol=REL_TOL,
    ), (
        f"{method}: Wrong complex variance. "
        f"Expected {expected_complex_var}, got {complex_var}"
    )

    print()
    print("PASSED")


def main():

    # Fixed seed -> reproducible test.
    torch.manual_seed(42)

    negative_slope = 0.01

    initializations = [
        {
            "method": "componentwise_pytorch_default",
        },
        {
            "method": "split_relu_he_normal",
        },
        {
            "method": "split_relu_he_uniform",
        },
        {
            "method": "split_leakyrelu_he_normal",
            "negative_slope": negative_slope,
        },
        {
            "method": "split_leakyrelu_he_uniform",
            "negative_slope": negative_slope,
        },
        {
            "method": "split_relu_xavier_normal",
        },
        {
            "method": "split_relu_xavier_uniform",
        },
        {
            "method": "split_leakyrelu_xavier_normal",
            "negative_slope": negative_slope,
        },
        {
            "method": "split_leakyrelu_xavier_uniform",
            "negative_slope": negative_slope,
        },
    ]

    for initialization in initializations:
        test_initialization(initialization)

    print()
    print("=" * 70)
    print("ALL INITIALIZATION TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()