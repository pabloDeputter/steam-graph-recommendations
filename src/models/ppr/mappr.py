from typing import Literal

import numpy as np
import torch

from src.models.ppr.ppr import PPR


class MultiAlphaPPR(PPR):
    """
    Multi-Alpha Personalized PageRank (PPR) recommendation model.

    This variant runs multiple PPR calculations with different alpha values and combines the results.
    Different alpha values allow capturing both short-range (high alpha) and long-range (low alpha)
    relationships in the graph simultaneously.

    :param alphas: List of alpha values to use for different PPR calculations
    :param alpha_weights: Optional weights for combining different alpha results, defaults to uniform weights
    :param num_iterations: Number of power iterations for PPR calculation
    :param popularity_weight: Base weight given to item popularity when combining with PPR scores
    :param interaction_weight_processing: Method for processing interaction weights
    :param continuation_strategy: Strategy used to derive node-dependent continuation probabilities
    :param exploration_bias: Scaling factor controlling how aggressively the continuation probabilities adapt
    """

    def __init__(
        self,
        alphas: list[float] = None,
        alpha_weights: list[float] | None = None,
        num_iterations: int = 20,
        popularity_weight: float = 0.2,
        interaction_weight_processing: Literal["log", "relative"] | None = "log",
        batch_size: int = 1024,
        continuation_strategy: Literal["fixed", "popularity_adaptive"] = "popularity_adaptive",
        exploration_bias: float = 0.25,
    ):
        # Validate alpha parameters
        if alphas is None:
            alphas = [0.25, 0.75, 0.85]
        if not alphas:
            raise ValueError("Must provide at least one alpha value")
        if not all(0 <= a <= 1 for a in alphas):
            raise ValueError("All alpha values must be between 0 and 1")

        # Set default uniform weights if none provided
        if alpha_weights is None:
            alpha_weights = [1.0 / len(alphas)] * len(alphas)

        # Validate weights
        if len(alphas) != len(alpha_weights):
            raise ValueError("Number of alphas must match number of weights")
        if not np.isclose(sum(alpha_weights), 1.0):
            raise ValueError("Alpha weights must sum to 1")

        # Initialize base PPR with dummy alpha (will be overridden)
        super().__init__(
            alpha=alphas[0],  # Temporary alpha, not actually used
            num_iterations=num_iterations,
            popularity_weight=popularity_weight,
            interaction_weight_processing=interaction_weight_processing,
            batch_size=batch_size,
            continuation_strategy=continuation_strategy,
            exploration_bias=exploration_bias,
        )

        self.alphas = alphas
        self.alpha_weights = alpha_weights

    def _calculatePPR(self, personalization_vector: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined PPR scores using multiple alpha values.

        For each alpha value, calculates a separate PPR score and combines them using
        the specified weights.

        :param personalization_vector: Vector representing user's historical interactions

        :return: Combined PPR scores for all items
        """
        if personalization_vector.ndim == 1:
            personalization = personalization_vector.unsqueeze(0)
            squeeze_output = True
        else:
            personalization = personalization_vector
            squeeze_output = False

        final_ppr = torch.zeros_like(personalization)

        # Calculate PPR for each alpha and combine weighted results
        for alpha, weight in zip(self.alphas, self.alpha_weights, strict=False):
            # Initialize PPR scores with personalization vector
            ppr = personalization.clone()
            continuation = self._getContinuationForAlpha(alpha).unsqueeze(0)

            # Power iteration for current alpha
            for _ in range(self.num_iterations):
                walk_scores = ppr @ self._iu_matrix @ self._ui_matrix
                ppr_next = continuation * walk_scores + (1 - continuation) * personalization

                # Check convergence
                if torch.allclose(ppr, ppr_next, atol=1e-7):
                    self.logger.info(f"Converged after {_ + 1} iterations for alpha {alpha}")
                    ppr = ppr_next
                    break
                ppr = ppr_next

            # Add weighted contribution from current alpha
            final_ppr += weight * ppr

        if squeeze_output:
            return final_ppr.squeeze(0)
        return final_ppr
