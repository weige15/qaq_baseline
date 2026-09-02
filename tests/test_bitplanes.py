import unittest

import numpy as np

from qaq.bitplanes import (
    from_sign_magnitude_planes,
    symmetric_dequantize,
    symmetric_quantize,
    to_sign_magnitude_planes,
)


class SignMagnitudePlaneTests(unittest.TestCase):
    def test_full_width_round_trip_is_exact(self) -> None:
        values = np.arange(-127, 128, dtype=np.int32)
        planes = to_sign_magnitude_planes(values, bits=8)
        rebuilt = from_sign_magnitude_planes(planes, precision=8)
        np.testing.assert_array_equal(rebuilt, values)

    def test_more_planes_never_increase_integer_error(self) -> None:
        values = np.array([-127, -101, -63, -17, -1, 0, 1, 17, 63, 101, 127])
        planes = to_sign_magnitude_planes(values, bits=8)
        previous = None
        for precision in range(2, 9):
            rebuilt = from_sign_magnitude_planes(planes, precision=precision)
            error = np.abs(values - rebuilt)
            if previous is not None:
                self.assertTrue(np.all(error <= previous))
            previous = error

    def test_discarded_lower_planes_are_zero(self) -> None:
        values = np.array([-127, -65, -31, 31, 65, 127])
        planes = to_sign_magnitude_planes(values, bits=8)
        for precision in range(2, 8):
            rebuilt = from_sign_magnitude_planes(planes, precision=precision)
            quantum = 1 << (8 - precision)
            self.assertTrue(np.all(np.abs(rebuilt) % quantum == 0))

    def test_symmetric_quantization_has_half_step_bound(self) -> None:
        weights = np.array(
            [[-1.0, -0.25, 0.0, 0.24, 1.0], [-3.0, -1.2, 0.2, 1.7, 2.9]],
            dtype=np.float64,
        )
        quantized = symmetric_quantize(weights, bits=8, reduce_axis=-1)
        recovered = symmetric_dequantize(quantized)
        error = np.abs(weights - recovered)
        self.assertTrue(np.all(error <= quantized.scale / 2 + 1e-12))

    def test_zero_rows_remain_zero(self) -> None:
        weights = np.zeros((2, 4), dtype=np.float64)
        quantized = symmetric_quantize(weights, bits=8, reduce_axis=-1)
        np.testing.assert_array_equal(quantized.values, np.zeros((2, 4), dtype=np.int32))
        np.testing.assert_array_equal(symmetric_dequantize(quantized), weights)

    def test_invalid_precision_is_rejected(self) -> None:
        planes = to_sign_magnitude_planes(np.array([1, -1]), bits=8)
        with self.assertRaises(ValueError):
            from_sign_magnitude_planes(planes, precision=1)
        with self.assertRaises(ValueError):
            from_sign_magnitude_planes(planes, precision=9)


if __name__ == "__main__":
    unittest.main()

