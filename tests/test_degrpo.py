# tests/test_degrpo.py

from trace_core.degrpo import calculate_degrpo_loss
import tensorflow as tf
import pytest  # Pytest is the framework we use for testing


def test_loss_function_returns_scalar_tensor():
    """
    A unit test to guarantee the loss function always returns a single value.

    The final output of any loss function must be a scalar (a single number),
    because this is what the optimizer needs to work. This test checks for that.
    """
    # 1. ARRANGE: Create dummy input data that looks like real data.
    # A batch of 2 states, with logits for 2 possible actions.
    dummy_logits = tf.constant([[0.1, 0.9], [0.8, 0.2]], dtype=tf.float32)
    dummy_advantages = tf.constant([1.0, -1.0], dtype=tf.float32)
    beta = 0.1

    # 2. ACT: Call the function we are testing.
    loss_value = calculate_degrpo_loss(dummy_logits, dummy_advantages, beta)

    # 3. ASSERT: Check that the result is what we expect.
    # We check that the result is a TensorFlow tensor.
    assert isinstance(loss_value, tf.Tensor)
    # We check that the tensor has rank 0, which means it's a scalar.
    assert tf.rank(loss_value) == 0, "Loss must be a scalar value."
