from typing import Literal

import torch

from src.models.ppr.ppr import PPR


class TwoPhasePPR(PPR):
    """
    Two-Phase Personalized PageRank (PPR) recommendation model.

    This variant performs PPR in two phases:
    1. Broad exploration phase with lower alpha to discover potential candidates
    2. Focused exploration phase with higher alpha on the top candidates from phase 1

    This approach helps balance between exploration and exploitation in the recommendation process.

    :param alpha1: Alpha value for first phase (broad exploration)
    :param alpha2: Alpha value for second phase (focused exploration)
    :param stage1_k: Number of top items to consider from first stage
    :param num_iterations: Number of power iterations for PPR calculation
    :param popularity_weight: Base weight given to item popularity when combining with PPR scores
    :param interaction_weight_processing: Method for processing interaction weights
    :param continuation_strategy: Strategy used to derive node-dependent continuation probabilities
    :param exploration_bias: Scaling factor controlling adaptive exploration
    """

    def __init__(
        self,
        alpha1: float = 0.3,
        alpha2: float = 0.7,
        stage1_k: int = 100,
        num_iterations: int = 100,
        popularity_weight: float = 0.2,
        interaction_weight_processing: str | None = "log",
        batch_size: int = 1024,
        continuation_strategy: Literal["fixed", "popularity_adaptive"] = "popularity_adaptive",
        exploration_bias: float = 0.25,
    ):
        # Validate parameters
        if not 0 <= alpha1 <= 1:
            raise ValueError(f"alpha1 must be between 0 and 1, got {alpha1}")
        if not 0 <= alpha2 <= 1:
            raise ValueError(f"alpha2 must be between 0 and 1, got {alpha2}")
        if stage1_k <= 0:
            raise ValueError(f"stage1_k must be positive, got {stage1_k}")

        # Initialize base PPR with dummy alpha (will be overridden in _calculatePPR)
        super().__init__(
            alpha=alpha1,  # Temporary alpha, not actually used
            num_iterations=num_iterations,
            popularity_weight=popularity_weight,
            interaction_weight_processing=interaction_weight_processing,
            batch_size=batch_size,
            continuation_strategy=continuation_strategy,
            exploration_bias=exploration_bias,
        )

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.stage1_k = stage1_k

    def _calculatePPR(self, personalization_vector: torch.Tensor) -> torch.Tensor:
        """
        Calculate PPR scores using two-phase approach.

        Phase 1: Broad exploration with lower alpha
        Phase 2: Focused exploration on top-k items with higher alpha

        :param personalization_vector: Vector representing user's historical interactions
        :return: Combined PPR scores from both phases
        """
        if personalization_vector.ndim == 1:
            personalization = personalization_vector.unsqueeze(0)
            squeeze_output = True
        else:
            personalization = personalization_vector
            squeeze_output = False

        # Phase 1: Broad exploration with lower alpha
        ppr1 = personalization.clone()
        continuation1 = self._getContinuationForAlpha(self.alpha1).unsqueeze(0)

        for _ in range(self.num_iterations):
            walk_scores = ppr1 @ self._iu_matrix @ self._ui_matrix
            ppr1_next = continuation1 * walk_scores + (1 - continuation1) * personalization

            if torch.allclose(ppr1, ppr1_next, atol=1e-7):
                self.logger.info(f"Converged after {_ + 1} iterations")
                ppr1 = ppr1_next
                break
            ppr1 = ppr1_next

        # Get top-k items from first phase
        _, top_k_indices = torch.topk(ppr1, min(self.stage1_k, ppr1.size(-1)), dim=-1)

        # Create new personalization vector for phase 2
        personalization2 = torch.zeros_like(personalization)
        top_k_actual = top_k_indices.size(-1)
        if top_k_actual == 0:
            raise ValueError("stage1_k selection returned zero candidates. Consider increasing stage1_k.")
        personalization2.scatter_(-1, top_k_indices, 1.0 / top_k_actual)

        # Phase 2: Focused exploration with higher alpha
        ppr2 = personalization2.clone()
        continuation2 = self._getContinuationForAlpha(self.alpha2).unsqueeze(0)

        for _ in range(self.num_iterations):
            walk_scores = ppr2 @ self._iu_matrix @ self._ui_matrix
            ppr2_next = continuation2 * walk_scores + (1 - continuation2) * personalization2

            if torch.allclose(ppr2, ppr2_next, atol=1e-7):
                self.logger.info(f"Converged after {_ + 1} iterations")
                ppr2 = ppr2_next
                break
            ppr2 = ppr2_next

        # Combine results from both phases (average)
        final_ppr = (ppr1 + ppr2) / 2

        # Sanity check: ensure output dimensions match input
        assert final_ppr.shape == personalization.shape, (
            f"PPR shape mismatch: expected {personalization.shape}, got {final_ppr.shape}"
        )

        if squeeze_output:
            return final_ppr.squeeze(0)
        return final_ppr
