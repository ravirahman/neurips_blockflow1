import numpy as np

class DPBase:
    def perturb_batch_input(self, num_epochs: int, x_s: np.ndarray) -> None:
        # Will be called on a batch, before training. Modifies x_s in place.
        pass

    def perturb_round_weights(self, num_unique_records: int, weights: np.ndarray) -> None:
        # Will be called once per round, after training and setting the weights in the model. Modifies weight in place.
        pass
