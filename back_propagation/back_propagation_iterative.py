import logging

import numpy as np
import numpy.typing as npt

# initial parameters
inputs = np.array([1, 0])
y_expected = 0
weights_hidden = np.array([[0.2, 0.4, 0.7, 0.5], [0.3, 0.5, 0.6, 0.9]])
weights_output = np.array([0.2, 0.4, 0.6, 0.8])
bias_hidden = 1
bias_output = 1
learning_step = 0.8

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

WeightsMatrix = npt.NDArray[np.float64]
WeightsVector = npt.NDArray[np.float64]


def train_neural_network_using_back_propagation(
        inputs: npt.NDArray[np.float64],  # (2,)
        weights_hidden: WeightsMatrix,  # (2, 4)
        weights_output: WeightsVector,  # (4,)
        bias_hidden: float,
        bias_output: float,
        y_expected: float,
        learning_step: float  # < 1
) -> (WeightsMatrix, WeightsVector, float, float, float):
    # FORWARD FEED

    # Z hidden
    Z_hidden = np.dot(inputs, weights_hidden) + bias_hidden
    logger.debug(f"Z hidden: {Z_hidden}")

    # activation
    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    y_hidden = sigmoid(Z_hidden)
    logger.debug(f"y hidden: {y_hidden}")

    # Z output

    Z_output = np.sum(y_hidden * weights_output) + bias_output  # one output neuron
    logger.debug(f"Z output: {Z_output}")

    # y output
    y_output = sigmoid(Z_output)
    logger.debug(f"y output: {y_output}")

    # squared error
    E = pow(y_expected - y_output, 2) / 2
    logger.debug(f"Squared error: {E}")

    # OUTPUT LAYER ADJUSTMENT

    # weights gradient
    dE_to_dY_output = -(y_expected - y_output)
    dY_output_to_dZ_output = y_output * (1 - y_output)
    dZ_output_to_dW_output = y_hidden

    weights_output_gradient = dE_to_dY_output * dY_output_to_dZ_output * dZ_output_to_dW_output
    logger.debug(f"Gradient of the error with respect to hidden-output weights: {weights_output_gradient}", )

    # bias gradient
    dZ_output_to_dB_output = 1  # linear dependency (Z directly increases by b) TODO: check
    # bias derivative does not impact result - added only for clarity
    bias_output_gradient = dE_to_dY_output * dY_output_to_dZ_output * dZ_output_to_dB_output
    logger.debug(f"Gradient of the error with respect to hidden-output bias: {bias_output_gradient}")

    # weights adjustment
    weights_output_adjusted = weights_output - learning_step * weights_output_gradient
    logger.debug(f"Adjusted hidden-output weights: {weights_output_adjusted}")

    # bias adjustment
    bias_output_adjusted = bias_output - learning_step * bias_output_gradient
    logger.debug(f"Adjusted hidden-output bias: {bias_output_adjusted}")

    # HIDDEN LAYER ADJUSTMENT

    # weights gradient
    # TODO: what to use: weights_input vs weights_output?
    # dZ_output_to_dY_hidden = weights_input  # (2, 4)
    dZ_output_to_dY_hidden = weights_output  # (4, 1)

    dE_to_dY_hidden = dE_to_dY_output * dY_output_to_dZ_output * dZ_output_to_dY_hidden
    dY_hidden_to_dZ_hidden = y_hidden * (1 - y_hidden)  # (4, 1)
    dZ_hidden_to_dW_hidden = inputs  # (2, 1)

    # dE/dw_hidden = dE/dY_hidden * dY_hidden/dZ_hidden * dZ_hidden/dW_hidden
    # inputs have different shape than other components, so we need to multiply each input by each element of (4, 1)
    # np.outer(A, B) - multiply each element of A by each element of B
    weights_hidden_gradient = np.outer(dZ_hidden_to_dW_hidden, dE_to_dY_hidden * dY_hidden_to_dZ_hidden)
    logger.debug(f"Gradient of the error with respect to input-hidden weights: {weights_hidden_gradient}")

    # bias gradient
    dZ_hidden_to_dB_hidden = 1 # linear dependency (Z directly increases by b) TODO: check

    # dB does not impact result - added only for clarity
    bias_hidden_gradient = dE_to_dY_hidden * dY_hidden_to_dZ_hidden * dZ_hidden_to_dB_hidden
    logger.debug(f"Gradient of the error with respect to hidden bias: {bias_hidden_gradient}")

    # weights adjustment
    weights_hidden_adjusted = weights_hidden - learning_step * weights_hidden_gradient
    logger.debug(f"Adjusted input-hidden weights: {weights_hidden_adjusted}")

    # bias adjustment
    # TODO: check if sum is right here
    bias_hidden_cumulative_gradient = bias_hidden_gradient.sum()
    bias_hidden_adjusted = bias_hidden - learning_step * bias_hidden_cumulative_gradient
    logger.debug(f"Adjusted hidden bias: {bias_hidden_adjusted}")

    return weights_hidden_adjusted, weights_output_adjusted, bias_hidden_adjusted, bias_output_adjusted, y_output, E


logger.info("E - error between expected and actual")
for i in range(100):
    logger.info(f"----------ITERATION {i}----------")
    adjusted_values = train_neural_network_using_back_propagation(
        inputs,
        weights_hidden,
        weights_output,
        bias_hidden,
        bias_output,
        y_expected,
        learning_step
    )

    weights_hidden, weights_output, bias_hidden, bias_output, y, E = adjusted_values
    logger.info(f"inputs {inputs} lead to output {y}")
    logger.info(f"E = {E} on iteration {i}")
