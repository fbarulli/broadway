"""Split-conformal prediction-interval calibration.

The calibrated artifact name is owned here: the training step writes it next to
the training result and the evaluate step writes it next to the evaluation
result, so one constant keeps both consumers on the same file name.
"""

CALIBRATION_FILE = "calibration.json"
