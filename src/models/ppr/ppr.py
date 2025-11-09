from typing import Literal

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

from src.models.recommender import Recommender


class PPR(Recommender):
    """
    Base class for Personalized PageRank (PPR) based recommendation models.

    Constructs a user-item interaction graph and iteratively calculates PPR scores to determine item relevance
    for each user. The final recommendation scores are a combination of the calculated PPR scores and item
    popularity, adaptively weighted based on the user's interaction history length. Inspired by recent research on
    node-dependent restart schemes, the recommender can optionally modulate the random walk continuation
    probability for every item to bias the walk toward novel (long-tail) content without discarding exploitation
    signals from popular titles.

    :param alpha: teleport probability, higher values emphasize exploration, lower values emphasize exploitation
    :param num_iterations: number of power iterations, more iterations lead to better approximation
    :param popularity_weight: base weight given to item popularity when combining PPR scores with popularity
    :param interaction_weight_processing: method for processing interaction weights (e.g., playtime)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        num_iterations: int = 20,
        popularity_weight: float = 0.2,
        interaction_weight_processing: Literal["log", "relative"] | None = "log",
        batch_size: int = 1024,
        continuation_strategy: Literal["fixed", "popularity_adaptive"] = "popularity_adaptive",
        exploration_bias: float = 0.25,
    ):
        # Validate initialization parameters
        self._validateInitParameters(
            alpha,
            num_iterations,
            popularity_weight,
            interaction_weight_processing,
            continuation_strategy,
            exploration_bias,
        )

        super().__init__()

        self.alpha = alpha
        self.num_iterations = num_iterations
        self.popularity_weight = popularity_weight
        self.interaction_weight_processing = interaction_weight_processing
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.continuation_strategy = continuation_strategy
        self.exploration_bias = exploration_bias

        # Initialize internal mappings and matrices; these will be populated during the fit method
        # Maps original item IDs to internal integer IDs
        self._item_mapping: dict[int | str, int] | None = None
        # Maps internal integer IDs back to original item IDs
        self._reverse_item_mapping: dict[int, int | str] | None = None
        # Maps original user IDs to internal integer IDs
        self._user_mapping: dict[int | str, int] | None = None
        # Normalized user-item interaction matrix (for transitions)
        self._ui_matrix: torch.Tensor | None = None
        # Normalized item-user interaction matrix (for reverse transitions)
        self._iu_matrix: torch.Tensor | None = None
        # Precomputed item popularity scores
        self._item_popularity: torch.Tensor | None = None
        # Node-dependent continuation probabilities used in adaptive restart PPR updates
        self._continuation_probabilities: torch.Tensor | None = None

    @staticmethod
    def _validateInitParameters(
        alpha: float,
        num_iterations: int,
        popularity_weight: float,
        interaction_weight_processing: str | None,
        continuation_strategy: str,
        exploration_bias: float,
    ) -> None:
        """
        Validates initialization parameters for the PPR recommender.

        :param alpha: teleport probability
        :param num_iterations: number of power iterations
        :param popularity_weight: base popularity weight
        :param interaction_weight_processing: interaction weight processing method
        :param continuation_strategy: strategy used to derive node-dependent continuation probabilities
        :param exploration_bias: scaling factor used by adaptive continuation strategies
        """
        # Validate parameters
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be between 0 and 1, but got {alpha}.")
        if not 0 <= popularity_weight <= 1:
            raise ValueError(f"popularity_weight must be between 0 and 1, but got {popularity_weight}.")
        if num_iterations <= 0:
            raise ValueError(f"num_iterations must be positive, but got {num_iterations}.")
        if interaction_weight_processing not in ["log", "relative", None]:
            raise ValueError(
                f"interaction_weight_processing must be 'log' or 'relative' or 'None', but got {interaction_weight_processing}."
            )
        if continuation_strategy not in {"fixed", "popularity_adaptive"}:
            raise ValueError(
                "continuation_strategy must be 'fixed' or 'popularity_adaptive', "
                f"but got {continuation_strategy}."
            )
        if exploration_bias < 0:
            raise ValueError(f"exploration_bias must be non-negative, but got {exploration_bias}.")

    def _processInteractionWeights(self, df: pd.DataFrame, user_id_column: str, interaction_column: str) -> np.ndarray:
        """
        Process the interaction weights (e.g., playtime) based on the specified method.

        :param df: DataFrame containing user-item interactions with the interaction column
        :param user_id_column: column containing user IDs
        :param interaction_column: column containing interaction weights (e.g., playtime)

        :return: processed interaction weights
        """
        # Sanity check: ensure the interaction column exists in the DataFrame
        if interaction_column not in df.columns:
            raise ValueError(f"Interaction column '{interaction_column}' not found in DataFrame.")

        # Use raw interaction values if no processing is specified
        weights = df[interaction_column].values

        if self.interaction_weight_processing == "log":
            # Apply log transformation to dampen the effect of large values
            weights = np.log1p(weights)
            # Scale the log-transformed values between 0 and 1
            weights = MinMaxScaler().fit_transform(weights.reshape(-1, 1)).ravel()
        elif self.interaction_weight_processing == "relative":
            # Scale interaction weights relative to each user's average interaction
            weights = (
                df.groupby(user_id_column)[interaction_column].transform(lambda x: x / x.mean() or 1.0).values
            )  # Avoid division by zero

        # Sanity check: ensure that no NaN values are introduced during processing
        assert not np.isnan(weights).any(), "NaN values found in processed interaction weights."

        return weights

    def _createMappings(self, df: pd.DataFrame, user_id_column: str, item_id_column: str):
        """
        Create mappings from original user and item IDs to internal integer IDs.

        :param df: DataFrame containing user-item interactions
        :param user_id_column: column containing user IDs
        :param item_id_column: column containing item IDs
        """
        # Sanity check: ensure that user and item ID columns exist in the DataFrame
        for col in [user_id_column, item_id_column]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

        # Create user mapping
        self._user_mapping = {uid: idx for idx, uid in enumerate(df[user_id_column].unique())}

        # Create item mappings
        self._item_mapping = {iid: idx for idx, iid in enumerate(df[item_id_column].unique())}
        self._reverse_item_mapping = {idx: iid for iid, idx in self._item_mapping.items()}

        # Sanity check: ensure that there are users and items in the DataFrame
        num_users = len(self._user_mapping)
        num_items = len(self._item_mapping)
        assert num_users > 0, "No users found in the DataFrame."
        assert num_items > 0, "No items found in the DataFrame."

        # Sanity check: ensure that reverse mappings are consistent
        assert len(self._item_mapping) == len(self._reverse_item_mapping), "Item mappings are inconsistent."
        assert len(self._user_mapping) == len(df[user_id_column].unique()), "User mappings are inconsistent."

    def _createUIMatrix(
        self, df: pd.DataFrame, user_id_column: str, item_id_column: str, interaction_column: str
    ) -> csr_matrix:
        """
        Creates a sparse user-item interaction matrix.

        :param df: DataFrame containing user-item interactions
        :param user_id_column: column containing user IDs
        :param item_id_column: column containing item IDs

        :return: sparse user-item interaction matrix
        """
        # Sanity check: ensure that required columns exist in the DataFrame
        for col in [user_id_column, item_id_column, interaction_column]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

        # Map user and item IDs to their internal indices
        rows = [self._user_mapping[uid] for uid in df[user_id_column]]
        cols = [self._item_mapping[iid] for iid in df[item_id_column]]

        # Get the processed interaction weights
        weights = self._processInteractionWeights(df, user_id_column, interaction_column)

        # Create the sparse matrix in CSR format
        ui_matrix = csr_matrix(
            (weights, (rows, cols)),
            shape=(len(self._user_mapping), len(self._item_mapping)),
        )
        return ui_matrix

    def _computeContinuationProbabilities(self, log_popularity: np.ndarray) -> torch.Tensor:
        """
        Derive item-specific continuation probabilities for the adaptive restart PPR formulation.

        Recent work on node-dependent restart schemes (e.g., Benzi et al., 2015; Lee et al., 2023) shows that
        adapting restart probabilities according to node importance can substantially improve the discovery of
        under-exposed items. We follow this idea by constructing continuation probabilities that penalize very
        popular games and gently boost exploratory mass toward the long tail.

        :param log_popularity: log-scaled item popularity values computed during fitting

        :return: tensor containing continuation probabilities for every item
        """

        num_items = log_popularity.shape[0]
        if num_items == 0:
            raise ValueError("Continuation probabilities cannot be computed without items.")

        if self.continuation_strategy == "fixed":
            continuation = torch.full((num_items,), float(self.alpha), dtype=torch.float32)
        else:
            popularity_tensor = torch.from_numpy(log_popularity).float()

            mean = popularity_tensor.mean()
            std = popularity_tensor.std(unbiased=False)
            if std < 1e-8:
                normalized_popularity = torch.zeros_like(popularity_tensor)
            else:
                normalized_popularity = (popularity_tensor - mean) / (std + 1e-8)

            # Tanh keeps adjustments bounded while respecting the sign of the deviation from the mean popularity.
            adjustment = torch.tanh(normalized_popularity) * float(self.exploration_bias)

            # Encourage restarts on extremely popular items while giving long-tail items more opportunity to surface.
            continuation = torch.clamp(self.alpha - adjustment, min=1e-3, max=0.999)

            # Keep the global mean close to the original alpha to maintain comparable mixing behaviour.
            continuation = continuation - continuation.mean() + self.alpha
            continuation = torch.clamp(continuation, min=1e-3, max=0.999)

        return continuation.to(self.device)

    def _getContinuationForAlpha(self, target_alpha: float) -> torch.Tensor:
        """Project the base continuation probabilities onto a desired global alpha level."""

        if self._continuation_probabilities is None:
            raise ValueError("Continuation probabilities are undefined. Ensure fit() has been called.")

        continuation = self._continuation_probabilities

        adjusted = continuation + (float(target_alpha) - float(self.alpha))
        adjusted = adjusted - adjusted.mean() + float(target_alpha)
        return torch.clamp(adjusted, min=1e-3, max=0.999)

    @staticmethod
    def _normalizeMatrix(matrix: csr_matrix, axis: int, threshold: float = 1e-8) -> csr_matrix:
        """
        Normalizes a sparse matrix row-wise or column-wise.

        For each row/column:
            - If all elements are zero => keep all zeros
            - If at least one non-zero element exists => set the largest element to 1, others to 0

        :param matrix: sparse matrix to normalize
        :param axis: 1 for row-wise normalization, 0 for column-wise
        :param threshold: tolerance for checking if normalized values sum up to 1

        :return: normalized sparse matrix
        """
        if axis not in (0, 1):
            raise ValueError("Axis must be 0 (column-wise) or 1 (row-wise)")

        # Normalize row-wise
        if axis == 1:
            row_sums = np.array(matrix.sum(axis=1)).flatten()
            # Only replace sum with 1 if there are any valid elements in the row
            row_sums[(row_sums == 0) & (matrix.getnnz(axis=1) > 0)] = 1
            normalized = matrix.multiply(1 / row_sums[:, np.newaxis])

            # Sanity check: ensure normalization
            row_sums = normalized.sum(axis=1).A1
            non_zero_rows = row_sums != 0
            """
            print(f"\nRow-wise normalization check:")
            print(f"Total rows: {len(row_sums)}")
            print(f"Non-zero rows: {np.sum(non_zero_rows)}")
            print(f"Zero rows: {np.sum(~non_zero_rows)}")
            """

            # Check for rows that don't sum to 1 (within threshold)
            abs_error = np.abs(row_sums - 1)
            problematic_rows = (abs_error > threshold) & non_zero_rows
            if np.any(problematic_rows):
                problem_indices = np.where(problematic_rows)[0]
                raise AssertionError(
                    f"{np.sum(problematic_rows)} rows with indices {problem_indices} don't sum to 1 (threshold: {threshold})."  # noqa: E501
                )

        # Normalize column-wise
        else:
            col_sums = np.array(matrix.sum(axis=0)).flatten()
            # Only replace sum with 1 if there are any valid elements in the column
            col_sums[(col_sums == 0) & (matrix.getnnz(axis=0) > 0)] = 1
            normalized = matrix.multiply(1 / col_sums[np.newaxis, :])

            # Sanity check: ensure normalization
            col_sums = normalized.sum(axis=0).A1
            non_zero_cols = col_sums != 0
            """
            print(f"\nColumn-wise normalization check:")
            print(f"Total columns: {len(col_sums)}")
            print(f"Non-zero columns: {np.sum(non_zero_cols)}")
            print(f"Zero columns: {np.sum(~non_zero_cols)}")
            """

            # Check for columns that don't sum to 1 (within threshold)
            abs_error = np.abs(col_sums - 1)
            problematic_cols = (abs_error > threshold) & non_zero_cols
            if np.any(problematic_cols):
                problem_indices = np.where(problematic_cols)[0]
                raise AssertionError(
                    f"{np.sum(problematic_cols)} columns with indices {problem_indices} don't sum to 1 (threshold: {threshold})."  # noqa: E501
                )

        return normalized

    def _calculatePPR(self, personalization_vector: torch.Tensor) -> torch.Tensor:
        """
        Calculate Personalized PageRank (PPR) scores using the specified personalization vector.

        :param personalization_vector: vector representing the user's historical interactions, seed items

        :return: The calculated PPR scores for all items
        """
        if personalization_vector.ndim == 1:
            personalization = personalization_vector.unsqueeze(0)
            squeeze_output = True
        else:
            personalization = personalization_vector
            squeeze_output = False

        # Initialize PPR scores with the personalization vector
        ppr = personalization.clone()
        continuation = self._getContinuationForAlpha(self.alpha).unsqueeze(0)

        # Power iteration
        for _ in range(self.num_iterations):
            # Two-step random walk: item->user->item
            # Apply PPR update rule: PPR_next = alpha * (PPR * IU * UI) + (1 - alpha) * personalization_vector
            walk_scores = ppr @ self._iu_matrix @ self._ui_matrix
            ppr_next = continuation * walk_scores + (1 - continuation) * personalization

            # Sanity check: ensure that the PPR dimensions remain consistent
            assert ppr_next.shape == ppr.shape, f"PPR shape mismatch, expected {ppr.shape}, got {ppr_next.shape}."

            # Check convergence, by comparing ppr and ppr_next
            if torch.allclose(ppr, ppr_next, atol=1e-7):
                self.logger.info(f"PPR converged after {_ + 1} iterations.")
                ppr = ppr_next
                break
            ppr = ppr_next
        if squeeze_output:
            return ppr.squeeze(0)
        return ppr

    def _batchPPR(self, batch_seed_items: list[list[int]]) -> torch.Tensor:
        """
        Calculates PPR scores for a batch of users based on their seed items.

        Examples for different user histories with base popularity_weight=0.45:
            1. User with NO history (cold-start):
            - History length = 0
            - Adaptive weight = 0.9 (fixed high value)
            - Final score = 0.9 * popularity_score + 0.1 * ppr_score

            2. User with 1 item in history:
            - History length = 1
            - Extra weight = (5 - 1) * 0.1 = 0.4
            - Adaptive weight = min(0.8, 0.45 + 0.4) = 0.8
            - Final score = 0.8 * popularity_score + 0.2 * ppr_score

            3. User with 3 items:
            - History length = 3
            - Extra weight = (5 - 3) * 0.1 = 0.2
            - Adaptive weight = min(0.8, 0.45 + 0.2) = 0.65
            - Final score = 0.65 * popularity_score + 0.35 * ppr_score

            4. User with 5 items:
            - History length = 5
            - Extra weight = (5 - 5) * 0.1 = 0
            - Adaptive weight = min(0.8, 0.45 + 0) = 0.45
            - Final score = 0.45 * popularity_score + 0.55 * ppr_score

            5. User with 10 items:
            - History length > 5
            - No extra weight
            - Adaptive weight = 0.45 (base weight)
            - Final score = 0.45 * popularity_score + 0.55 * ppr_score

        :param batch_seed_items:  List of seed item indices for each user

        :return: A tensor of PPR scores for each user in the batch
        """
        num_items = len(self._item_mapping)
        batch_size = len(batch_seed_items)

        # Initialize personalization vectors for the batch
        personalization = torch.zeros((batch_size, num_items), device=self.device)

        # Sanity check: ensure that the personalization vector dimensions are correct
        assert personalization.shape == (batch_size, num_items), (
            f"Personalization vector shape mismatch, expected {(batch_size, num_items)}, got {personalization.shape}."
        )

        # Calculate adaptive popularity weights based on the number of seed items
        popularity_weights = torch.zeros(batch_size, device=self.device)

        for i, seed_items in enumerate(batch_seed_items):
            if seed_items:
                # Initialize personalization vector with uniform distribution over seed items
                personalization[i, seed_items] = 1.0 / len(seed_items)
                history_length = len(seed_items)
                # Adjust popularity weight based on history length
                if history_length <= 5:
                    popularity_weights[i] = min(0.5, self.popularity_weight + (5 - history_length) * 0.1)
                else:
                    # Use base popularity weight if history length exceeds 5
                    popularity_weights[i] = self.popularity_weight
            else:
                # If no history, use item popularity as the personalization vector
                personalization[i] = self._item_popularity
                # Higher weight for cold-start users
                popularity_weights[i] = 0.9

        # Calculate PPR scores for the entire batch
        ppr = self._calculatePPR(personalization)

        # Apply popularity weighting, reshape for broadcasting
        popularity_weights = popularity_weights.unsqueeze(1)

        return (1 - popularity_weights) * ppr + popularity_weights * self._batch_popularity

    def _processUserBatch(
        self,
        batch_users: list[int | str],
        user_items: dict[int | str, list[int | str]],
        n_items: int,
    ) -> dict[int | str, list[int | str]]:
        """
        Generates recommendations for a batch of users.

        :param batch_users: list of user IDs for the current batch
        :param user_items: dict mapping user IDS to their historical items interactions
        :param n_items: number of items to generate per user

        :return: dict of user IDs mapped to a list of recommended item IDs
        """
        # Initialize seed items for the batch
        batch_seed_items = []
        # Store historical items for masking
        batch_historical_items = []

        # Initialize seed items for each user in the batch
        for user_id in batch_users:
            if user_id in user_items:
                # Get indices of user's historical items
                historical_items = [self._item_mapping[i] for i in user_items[user_id] if i in self._item_mapping]
                batch_seed_items.append(historical_items if historical_items else [])
                batch_historical_items.append(set(historical_items))
            else:
                # If user has no history, use an empty seed list
                batch_seed_items.append([])
                batch_historical_items.append(set())

        # Calculate PPR scores for the entire batch
        with torch.no_grad():
            batch_scores = self._batchPPR(batch_seed_items)

        # Process recommendations for each user in the batch
        batch_recommendations = {}
        for i, user_id in enumerate(batch_users):
            # Clone to avoid modifying original scores
            scores = batch_scores[i].clone()

            # Mask out historical items with negative infinity
            if batch_historical_items[i]:
                historical_idx = list(batch_historical_items[i])
                scores[historical_idx] = float("-inf")

            # Get top items, excluding historical items
            top_item_idx = torch.topk(scores, min(n_items + len(batch_historical_items[i]), len(self._item_mapping)))[1]

            # Convert to list and filter out any historical items that might have slipped through
            recommended_items = []
            for idx in top_item_idx.cpu().numpy():
                item_id = self._reverse_item_mapping[idx]
                if user_id in user_items and item_id in user_items[user_id]:
                    continue
                recommended_items.append(item_id)
                if len(recommended_items) == n_items:
                    break

            batch_recommendations[user_id] = recommended_items[:n_items]

        assert len(batch_recommendations) == len(batch_users), f"Expected {len(batch_users)} recommendations."
        assert all(len(recs) == n_items for recs in batch_recommendations.values()), (
            f"Expected {n_items} recommendations per user."
        )

        return batch_recommendations

    def fit(
        self,
        df: pd.DataFrame,
        user_id_column: str = "user_id",
        item_id_column: str = "item_id",
        interaction_column: str = "playtime",
    ):
        """
        Fits the PPR model on the provided user-item interaction data.

        This method performs the following steps:
            1. Creates mappings for users and items to internal indices.
            2. Builds a sparse user-item interaction matrix from the data (interaction weights are processed here).
            3. Calculates item popularity scores.
            4. Builds the item-user matrix (transpose of user-item matrix).
            5. Normalizes the user-item and item-user matrices to obtain transition probabilities.

        :param df: DataFrame containing user-item interactions
        :param user_id_column: column containing user IDs
        :param item_id_column: column containing item IDs
        :param interaction_column: column containing interaction weights (e.g., playtime)
        """
        # Sanity check: ensure that required columns actually exist
        for col in [user_id_column, item_id_column, interaction_column]:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame.")

        # Create internal ID mappings for users and items
        self._createMappings(df, user_id_column, item_id_column)

        self.logger.info("Creating user-item interaction matrix")
        # Create sparse user-item interaction matrix (interaction weights are processed during creation)
        ui_matrix = self._createUIMatrix(df, user_id_column, item_id_column, interaction_column)

        # Calculate item popularity based on the sum of interactions
        raw_popularity = np.array(ui_matrix.sum(axis=0)).flatten()
        log_popularity = np.log1p(raw_popularity)
        # Dampen extreme values using log transformation
        if log_popularity.sum() == 0:
            popularity_distribution = np.ones_like(log_popularity) / len(log_popularity)
        else:
            popularity_distribution = log_popularity / log_popularity.sum()
        self._item_popularity = torch.from_numpy(popularity_distribution).float().to(self.device)

        # Compute continuation probabilities for adaptive restart behaviour
        self._continuation_probabilities = self._computeContinuationProbabilities(log_popularity)

        # Initialize batch popularity tensor for easy broadcasting
        self._batch_popularity = self._item_popularity.unsqueeze(0)

        # Normalize the user-item matrix row-wise to obtain transition probabilities
        ui_matrix_normalized = self._normalizeMatrix(ui_matrix, axis=1)

        self.logger.info("Creating item-user interaction matrix")
        # Transpose the user-item matrix to obtain the item-user matrix
        iu_matrix = ui_matrix_normalized.T

        # Normalize the item-user matrix column-wise to obtain reverse transition probabilities
        iu_matrix_normalized = self._normalizeMatrix(iu_matrix, axis=1)

        # Convert the normalized sparse matrix to a PyTorch tensor for efficient computation
        self._ui_matrix = torch.from_numpy(ui_matrix_normalized.toarray()).float().to(self.device)
        self._iu_matrix = torch.from_numpy(iu_matrix_normalized.toarray()).float().to(self.device)

        num_users = len(self._user_mapping)
        num_items = len(self._item_mapping)

        # Sanity checks: ensure the normalized matrices have the correct dimensions
        assert self._ui_matrix.shape == (num_users, num_items), f"Invalid UI matrix shape: {self._ui_matrix.shape}."
        assert self._iu_matrix.shape == (num_items, num_users), f"Invalid IU matrix shape: {self._iu_matrix.shape}."

        # Sanity check: ensure that the item popularity tensor has the correct shape
        assert self._item_popularity.shape == (num_items,), (
            f"Invalid item popularity shape: {self._item_popularity.shape}."
        )

    def predict(
        self,
        test_df: pd.DataFrame,
        user_id_column: str = "user_id",
        item_id_column: str = "item_id",
        n_items: int = 20,
    ) -> dict[int | str, list[int | str]]:
        """
        Generate recommendations for users in the provided test DataFrame.

        :param test_df: DataFrame containing user-item interactions for which recommendations are to be generated
        :param user_id_column: column containing user IDs
        :param item_id_column: column containing item IDs
        :param n_items: number of items to recommend per user
        :param batch_size: batch size for processing users

        :return: dictionary of user IDs mapped to a list of recommended item IDs
        """
        # Sanity check: ensure that the model has been fitted
        if (
            not hasattr(self, "_ui_matrix")
            or self._ui_matrix is None
            or not hasattr(self, "_iu_matrix")
            or self._iu_matrix is None
        ):
            raise ValueError("Model must be fitted before generating recommendations.")

        # Sanity check: ensure that required columns exist in the DataFrame
        for col in [user_id_column, item_id_column]:
            if col not in test_df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

        # Group test interactions by user to easily access historical items
        user_items = test_df.groupby(user_id_column)[item_id_column].agg(list).to_dict()
        unique_users = list(test_df[user_id_column].unique())
        total_users = len(unique_users)

        # Process in batches
        recommendations: dict[int | str, list[int | str]] = {}
        progress_bar = tqdm(range(0, total_users, self.batch_size), desc="Generating recommendations")

        for i in progress_bar:
            # Process a batch of users
            batch_users = unique_users[i : i + self.batch_size]
            batch_recommendations = self._processUserBatch(batch_users, user_items, n_items)
            # Update recommendations dictionary
            recommendations.update(batch_recommendations)

            # Update progress description
            progress_bar.set_description(f"Processed {min(i + self.batch_size, total_users)}/{total_users} users")
        return recommendations
