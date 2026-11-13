"""Greedy Selection of Independent labels for MLC with partial abstention.

GSI-MLC-PA combines the two MLP mechanisms already used by this project:

* BR-MLP estimates direct marginal probabilities for every label.
* CC-MLP estimates a label conditionally on its predecessors.

The independent/dependent partition is selected on an internal validation
split.  The partition is frozen before the two probability models are refit
on all outer-training data, so an outer test fold is never used for model
selection.
"""

from copy import deepcopy
from time import perf_counter

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
except ImportError:  # pragma: no cover - project requirements normally provide it
    MultilabelStratifiedShuffleSplit = None

from ..decision import canonical_policy_name, create_configured_policy
from ..selection import (
    canonical_final_order_strategy,
    canonical_partition_mode,
    canonical_selection_objective,
    evaluate_selection_objective,
    provide_partition,
)
from .base_learners import (
    base_learner_manifest,
    canonical_base_learner_name,
    create_binary_estimator,
    create_multilabel_estimator,
)
from .classifier_chain import ClassifierChainClassifier


def _clone_or_copy(estimator):
    try:
        return clone(estimator)
    except (TypeError, RuntimeError):
        return deepcopy(estimator)


def _as_dense_float(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix.")
    return X


def _positive_probability(classifier, X):
    """Return P(Y=1) from a binary classifier with a tolerant API adapter."""
    if hasattr(classifier, "predict_proba"):
        probabilities = np.asarray(classifier.predict_proba(X), dtype=np.float64)
        if probabilities.ndim == 1:
            result = probabilities
        elif probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            result = probabilities[:, 1]
        elif probabilities.ndim == 2 and probabilities.shape[1] == 1:
            result = probabilities[:, 0]
        else:
            raise ValueError("Binary predict_proba() returned an invalid shape.")
    elif hasattr(classifier, "decision_function"):
        scores = np.asarray(classifier.decision_function(X), dtype=np.float64)
        result = 1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0)))
    else:
        result = np.asarray(classifier.predict(X), dtype=np.float64)
    return np.clip(np.ravel(result), 0.0, 1.0)


class GSIMLCPartialAbstentionClassifier(BaseEstimator, ClassifierMixin):
    """GSI-MLC-PA with configurable leakage-safe IL/DL selection.

    Parameters
    ----------
    cost : float, default=0.3
        Default BOP rejection cost used by :meth:`predict` and by partial-BOP
        selection objectives. The compatible ``full_macro_f1`` default does
        not depend on this cost.
    validation_size : float, default=0.2
        Fraction of the outer-training fold used only for IL/DL selection.
    refit : bool, default=True
        Refit BR-MLP and CC-MLP on the complete outer-training fold after the
        partition has been frozen.
    order : sequence of int or None, default=None
        CC predecessor order.  None uses natural label order.
    br_estimator, cc_estimator : estimator or None
        Test/extensibility hooks.  Defaults use the same BR-MLP and CC-MLP
        implementations as the benchmark.
    abstain_value : int, default=-1
        Integer sentinel representing the abstention symbol.
    random_state : int, default=42
        Seed for the internal split and MLP estimators.
    selection_objective : str, default="full_macro_f1"
        Inner-validation objective used to accept or reject IL candidates.
    decision_policy : str, default="hamming"
        Final policy used by :meth:`predict`; selection objectives record their
        own corresponding policy separately.
    beta : float, default=1.0
        F-beta parameter used when the final decision policy is ``fbeta``.
    penalty : {"linear", "concave"}, default="linear"
        Abstention penalty used by partial selection and final policies.
    partition_mode : str, default="learned"
        IL/DL provider. Supports learned, learned without correlation reorder,
        all-IL, all-DL, fixed and random-matched ablations.
    partition_random_state : int or None, default=None
        Seed used only by ``random_matched``.  ``None`` preserves the legacy
        behavior by reusing ``random_state``.  A separate seed lets repeated
        random partitions share identical outer folds, inner splits and base
        learner fits.
    fixed_independent_labels : sequence of int or None, default=None
        IL indices required only when ``partition_mode="fixed"``.
    final_order : {"correlation", "selection", "natural"}, default="correlation"
        Order strategy applied after the partition is frozen. An explicit
        ``order`` permutation retains precedence for backward compatibility.
    base_learner : {"logistic", "mlp"}, default="mlp"
        Shared marginal/binary learner configuration used by the BR and CC
        components. Registered Q8 model IDs set this value explicitly.

    Notes
    -----
    For one predecessor, dependent inference computes the exact two-state
    marginal ``(1-p) f(x,0) + p f(x,1)``.  With multiple predecessors, it uses
    the standard mean-field plug-in approximation: each binary predecessor is
    replaced by its finalized soft marginal under the current IL/DL
    configuration before evaluating the conditional CC classifier.
    """

    def __init__(
        self,
        cost=0.3,
        validation_size=0.2,
        refit=True,
        order=None,
        br_estimator=None,
        cc_estimator=None,
        abstain_value=-1,
        random_state=42,
        selection_objective="full_macro_f1",
        decision_policy="hamming",
        beta=1.0,
        penalty="linear",
        partition_mode="learned",
        partition_random_state=None,
        fixed_independent_labels=None,
        final_order="correlation",
        base_learner="mlp",
        min_coverage=None,
        cc_label_noise=0.0,
        mlp_loss="bce",
        stratified_threshold=0.75,
        max_peeling_depth=3,
        dl_order_direction="ascending",
        use_complexity_penalty=False,
        complexity_penalty_lambda=0.01,
        complexity_penalty_beta=0.5,
        sparse_cc_threshold=None,
        aug_normalization="matching",
        decaying_threshold=False,
        threshold_decay_step=0.05,
        cv_folds=5,
        peeling_metric="selective_f1",
    ):
        self.cost = cost
        self.validation_size = validation_size
        self.refit = refit
        self.order = order
        self.br_estimator = br_estimator
        self.cc_estimator = cc_estimator
        self.abstain_value = abstain_value
        self.random_state = random_state
        self.selection_objective = selection_objective
        self.decision_policy = decision_policy
        self.beta = beta
        self.penalty = penalty
        self.partition_mode = partition_mode
        self.partition_random_state = partition_random_state
        self.fixed_independent_labels = fixed_independent_labels
        self.final_order = final_order
        self.base_learner = base_learner
        self.min_coverage = min_coverage
        self.cc_label_noise = cc_label_noise
        self.mlp_loss = mlp_loss
        self.stratified_threshold = float(stratified_threshold)
        self.max_peeling_depth = int(max_peeling_depth)
        self.dl_order_direction = str(dl_order_direction)
        self.use_complexity_penalty = bool(use_complexity_penalty)
        self.complexity_penalty_lambda = float(complexity_penalty_lambda)
        self.complexity_penalty_beta = float(complexity_penalty_beta)
        self.sparse_cc_threshold = (
            float(sparse_cc_threshold) if sparse_cc_threshold is not None else None
        )
        self.aug_normalization = str(aug_normalization)
        self.decaying_threshold = bool(decaying_threshold)
        self.threshold_decay_step = float(threshold_decay_step)
        self.cv_folds = int(cv_folds)
        self.peeling_metric = str(peeling_metric)

    def _validate_parameters(self):
        if not 0.0 <= float(self.cost) <= 1.0:
            raise ValueError("cost must lie in [0, 1].")
        if not 0.0 < float(self.validation_size) < 1.0:
            raise ValueError("validation_size must lie strictly between 0 and 1.")
        if self.abstain_value in (0, 1):
            raise ValueError("abstain_value must differ from 0 and 1.")
        beta = float(self.beta)
        if not np.isfinite(beta) or beta <= 0.0:
            raise ValueError("beta must be a finite positive number.")
        if not 0.0 <= float(self.stratified_threshold) <= 1.0:
            raise ValueError("stratified_threshold must lie in [0, 1].")
        if int(self.max_peeling_depth) < 1:
            raise ValueError("max_peeling_depth must be at least 1.")
        if int(self.cv_folds) < 2:
            raise ValueError("cv_folds must be at least 2.")
        if self.peeling_metric not in ("selective_f1", "optimal_f1", "standard_f1"):
            raise ValueError("peeling_metric must be 'selective_f1', 'optimal_f1', or 'standard_f1'.")
        if self.dl_order_direction not in ("ascending", "descending"):
            raise ValueError("dl_order_direction must be 'ascending' or 'descending'.")
        canonical_selection_objective(self.selection_objective)
        canonical_policy_name(self.decision_policy)
        canonical_partition_mode(self.partition_mode)
        if self.partition_random_state is not None and (
            isinstance(self.partition_random_state, (bool, np.bool_))
            or not isinstance(self.partition_random_state, (int, np.integer))
        ):
            raise ValueError("partition_random_state must be an integer or None.")
        canonical_final_order_strategy(self.final_order)
        canonical_base_learner_name(self.base_learner, default="mlp")
        self._make_decision_policy()

    def _validated_order(self, n_labels):
        if self.order is None:
            return list(range(n_labels))
        order = [int(label) for label in self.order]
        if sorted(order) != list(range(n_labels)):
            raise ValueError("order must be a permutation of all label indices.")
        return order

    def _make_br_model(self):
        if self.br_estimator is None:
            extra = {}
            if self.base_learner == "mlp" and self.mlp_loss != "bce":
                extra["loss"] = self.mlp_loss
            return create_multilabel_estimator(
                self.base_learner, random_state=self.random_state, **extra
            )
        return _clone_or_copy(self.br_estimator)

    def _make_cc_model(self, order=None):
        requested_order = self.order_ if order is None else list(order)
        corr_matrix = getattr(self, "label_correlation_", None)
        ind_labels = getattr(self, "independent_labels_", None)
        if self.cc_estimator is None:
            extra = {}
            if self.base_learner == "mlp" and self.mlp_loss != "bce":
                extra["loss"] = self.mlp_loss
            return ClassifierChainClassifier(
                base_estimator=create_binary_estimator(
                    self.base_learner, random_state=self.random_state, **extra
                ),
                order=requested_order,
                random_state=self.random_state,
                label_noise=self.cc_label_noise,
                correlation_matrix=corr_matrix,
                correlation_threshold=self.sparse_cc_threshold,
                independent_labels=ind_labels,
            )
        estimator = _clone_or_copy(self.cc_estimator)
        if hasattr(estimator, "set_params"):
            params_to_set = {"order": requested_order}
            if corr_matrix is not None:
                params_to_set["correlation_matrix"] = corr_matrix
            if self.sparse_cc_threshold is not None:
                params_to_set["correlation_threshold"] = self.sparse_cc_threshold
            if ind_labels is not None:
                params_to_set["independent_labels"] = ind_labels
            try:
                estimator.set_params(**params_to_set)
            except (TypeError, ValueError):
                try:
                    estimator.set_params(order=requested_order)
                except (TypeError, ValueError):
                    pass
        return estimator

    def _split_train_validation(self, X, Y):
        indices = np.arange(X.shape[0])
        if MultilabelStratifiedShuffleSplit is not None:
            try:
                splitter = MultilabelStratifiedShuffleSplit(
                    n_splits=1,
                    test_size=float(self.validation_size),
                    random_state=self.random_state,
                )
                train_indices, validation_indices = next(splitter.split(X, Y))
                return train_indices, validation_indices
            except (TypeError, ValueError):
                pass
        return train_test_split(
            indices,
            test_size=float(self.validation_size),
            random_state=self.random_state,
            shuffle=True,
        )

    @staticmethod
    def _validate_probability_matrix(probabilities, n_samples, n_labels, name):
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.shape != (n_samples, n_labels):
            raise ValueError(
                f"{name}.predict_proba() must return shape "
                f"({n_samples}, {n_labels}), got {probabilities.shape}."
            )
        return np.clip(probabilities, 0.0, 1.0)

    def _direct_probabilities(self, model, X):
        return self._validate_probability_matrix(
            model.predict_proba(X), X.shape[0], self.n_labels_, "BR model"
        )

    def _configured_probabilities(
        self,
        X,
        direct_probabilities,
        cc_model,
        independent_labels,
        initial_probabilities=None,
        start_position=0,
    ):
        """Infer probabilities sequentially under one fixed IL/DL partition."""
        if not hasattr(cc_model, "classifiers_") or not hasattr(cc_model, "order_"):
            raise ValueError(
                "cc_estimator must expose fitted classifiers_ and order_ attributes."
            )
        if list(cc_model.order_) != self.order_:
            raise ValueError("The fitted CC order does not match the GSI order.")
        if len(cc_model.classifiers_) != self.n_labels_:
            raise ValueError("The fitted CC must contain one classifier per label.")

        if not 0 <= int(start_position) < self.n_labels_:
            raise ValueError("start_position must index a label in the chain.")
        independent = set(int(label) for label in independent_labels)
        if initial_probabilities is None:
            probabilities = np.zeros_like(direct_probabilities, dtype=np.float64)
            start_position = 0
        else:
            probabilities = self._validate_probability_matrix(
                initial_probabilities,
                X.shape[0],
                self.n_labels_,
                "initial probabilities",
            ).copy()
        for position, label_index in enumerate(self.order_):
            if position < start_position:
                continue
            if label_index in independent:
                probabilities[:, label_index] = direct_probabilities[:, label_index]
                continue

            # Determine active predecessors for this label
            if hasattr(cc_model, "active_parents_map_") and label_index in cc_model.active_parents_map_:
                active_predecessors = cc_model.active_parents_map_[label_index]
            elif self.sparse_cc_threshold is not None and self.sparse_cc_threshold > 0.0:
                corr_mat = getattr(self, "label_correlation_", None)
                if corr_mat is not None:
                    active_predecessors = [
                        p for p in self.order_[:position]
                        if (p in independent) or (float(corr_mat[p, label_index]) >= self.sparse_cc_threshold)
                    ]
                else:
                    active_predecessors = self.order_[:position]
            else:
                active_predecessors = self.order_[:position]

            classifier = cc_model.classifiers_[position]
            if not active_predecessors:
                # A DL chain root or isolated DL node is the unconditional CC classifier.
                probabilities[:, label_index] = _positive_probability(
                    classifier, X
                )
                continue

            # These are the already-finalized probabilities under the current
            # configuration: BR for preceding IL labels and marginalized CC
            # probabilities for preceding DL labels.
            predecessor_probabilities = probabilities[:, active_predecessors]
            if len(active_predecessors) == 1:
                zeros = np.zeros((X.shape[0], 1), dtype=np.float32)
                ones = np.ones((X.shape[0], 1), dtype=np.float32)
                probability_given_zero = _positive_probability(
                    classifier, np.hstack((X, zeros))
                )
                probability_given_one = _positive_probability(
                    classifier, np.hstack((X, ones))
                )
                parent_probability = predecessor_probabilities[:, 0]
                probabilities[:, label_index] = (
                    (1.0 - parent_probability) * probability_given_zero
                    + parent_probability * probability_given_one
                )
            else:
                # Mean-field marginalization: q(Y_parents) is represented by
                # its factorized means, avoiding an exponential 2**m sum.
                mean_field_features = np.hstack(
                    (X, predecessor_probabilities.astype(np.float32))
                )
                probabilities[:, label_index] = _positive_probability(
                    classifier, mean_field_features
                )
        return np.clip(probabilities, 0.0, 1.0)

    @staticmethod
    def _compute_label_correlation(Y):
        """Return a finite Phi/Pearson matrix for binary label columns."""
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim != 2:
            raise ValueError("Y must be a two-dimensional label matrix.")
        if Y.shape[1] == 1:
            return np.ones((1, 1), dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = np.corrcoef(Y, rowvar=False)
        correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
        correlation = np.clip(correlation, -1.0, 1.0)
        np.fill_diagonal(correlation, 1.0)
        return correlation

    @staticmethod
    def _correlation_order(
        correlation,
        independent_labels,
        dependent_labels,
        direction="descending",
    ):
        """Order labels after IL/DL is frozen using absolute correlations.

        Independent labels form deterministic roots. If direction == 'ascending',
        remaining dependent labels are ordered by ascending cumulative correlation
        (historical v5.1 design requirement: least coupled labels first).
        If direction == 'descending', remaining dependent labels are appended by
        strongest connection to an already placed label (legacy behavior).
        """
        correlation = np.asarray(correlation, dtype=np.float64)
        independent = sorted(int(label) for label in independent_labels)
        remaining = set(int(label) for label in dependent_labels)

        if direction == "ascending":
            remaining_list = sorted(list(remaining))
            scores = {}
            for label in remaining_list:
                other_labels = [j for j in remaining_list if j != label]
                scores[label] = (
                    float(np.sum(np.abs(correlation[label, other_labels])))
                    if other_labels
                    else 0.0
                )
            ordered_dl = sorted(remaining_list, key=lambda l: (scores[l], l))
            return independent + ordered_dl

        order = list(independent)
        if not order and remaining:
            root = min(
                remaining,
                key=lambda label: (
                    -float(np.sum(np.abs(correlation[label]))),
                    label,
                ),
            )
            order.append(root)
            remaining.remove(root)
        while remaining:
            best = min(
                remaining,
                key=lambda label: (
                    -max(
                        (abs(float(correlation[label, parent])) for parent in order),
                        default=0.0,
                    ),
                    label,
                ),
            )
            order.append(best)
            remaining.remove(best)
        return order

    @staticmethod
    def _dependent_parent_map(correlation, order, dependent_labels):
        """Record the strongest preceding parent for every dependent label."""
        correlation = np.asarray(correlation, dtype=np.float64)
        dependent = set(int(label) for label in dependent_labels)
        parent_map = {}
        for position, label in enumerate(order):
            if label not in dependent:
                continue
            predecessors = order[:position]
            if not predecessors:
                parent_map[int(label)] = None
                continue
            parent_map[int(label)] = int(
                min(
                    predecessors,
                    key=lambda parent: (
                        -abs(float(correlation[label, parent])),
                        parent,
                    ),
                )
            )
        return parent_map

    def _apply_bop(self, probabilities, cost=None):
        """Apply BOP once, after all final probabilities have been produced."""
        policy = getattr(self, "decision_policy_", None)
        if policy is None:
            policy = self._make_decision_policy()
        return policy.predict_from_proba(
            probabilities, cost=cost
        )

    def _make_decision_policy(self):
        """Build the final configured policy without embedding its formula."""

        return create_configured_policy(
            self.decision_policy,
            cost=self.cost,
            penalty=self.penalty,
            beta=self.beta,
            allow_abstention=True,
            abstain_value=self.abstain_value,
            hamming_boundary="symmetric_thresholds",
            min_coverage=self.min_coverage,
        )

    def _evaluate_configuration(self, Y, probabilities):
        """Delegate one candidate to the selection-objective registry."""

        return evaluate_selection_objective(
            self.selection_objective,
            Y,
            probabilities,
            cost=self.cost,
            penalty=self.penalty,
            abstain_value=self.abstain_value,
        )

    def _configuration_objective(self, Y, probabilities):
        """Compatibility scalar API backed by the objective registry."""

        return float(self._evaluate_configuration(Y, probabilities).score)

    def _selection_history_record(
        self,
        step,
        label,
        accepted,
        result,
        improvement,
    ):
        metadata = result.history_metadata()
        return {
            "step": int(step),
            "tested_label": None if label is None else int(label),
            "accepted": bool(accepted),
            "score": float(result.score),
            "full_macro_f1": float(result.diagnostics["full_macro_f1"]),
            "improvement": float(improvement),
            **metadata,
            "configured_cost": float(self.cost),
            "penalty": self.penalty,
            "inner_split_seed": int(self.random_state),
            "final_decision_policy": canonical_policy_name(self.decision_policy),
            "partition_mode": canonical_partition_mode(self.partition_mode),
        }

    def _store_selection_result(self, result):
        self.selection_objective_name_ = result.objective
        self.selection_policy_config_ = dict(result.policy_config)
        self.validation_objective_details_ = dict(result.diagnostics)
        self.validation_full_macro_f1_ = float(
            result.diagnostics["full_macro_f1"]
        )
        self.decision_policy_name_ = canonical_policy_name(self.decision_policy)
        self.decision_policy_config_ = self._make_decision_policy().get_config()
        self.selection_config_ = {
            "selection_objective": self.selection_objective_name_,
            "selection_policy": dict(self.selection_policy_config_),
            "final_decision_policy": dict(self.decision_policy_config_),
            "beta": None if result.beta is None else float(result.beta),
            "final_decision_beta": float(self.beta),
            "cost": float(self.cost),
            "penalty": self.penalty,
            "inner_split_seed": int(self.random_state),
        }

    def _score_frozen_partition(
        self,
        X,
        Y,
        direct_probabilities,
        cc_model,
        independent_labels,
    ):
        """Score a provider partition through the same Q6 objective API."""

        probabilities = self._configured_probabilities(
            X,
            direct_probabilities,
            cc_model,
            independent_labels,
        )
        result = self._evaluate_configuration(Y, probabilities)
        self.evaluated_configurations_ = 1
        self._store_selection_result(result)
        history = [
            self._selection_history_record(0, None, True, result, 0.0)
        ]
        return float(result.score), history

    def _select_partition(self, X, Y, direct_probabilities, cc_model):
        """Move labels from DL to IL using the configured objective module."""
        independent = []
        dependent = set(range(self.n_labels_))
        probability_cache = {}

        def evaluate(label_set, initial_probabilities=None, start_position=0):
            key = frozenset(label_set)
            if key not in probability_cache:
                candidate_probabilities = self._configured_probabilities(
                    X,
                    direct_probabilities,
                    cc_model,
                    key,
                    initial_probabilities=initial_probabilities,
                    start_position=start_position,
                )
                probability_cache[key] = (
                    candidate_probabilities,
                    self._evaluate_configuration(Y, candidate_probabilities),
                )
            return probability_cache[key]

        current_probabilities, current_result = evaluate(independent)
        current_score = float(current_result.score)
        history = [
            self._selection_history_record(0, None, True, current_result, 0.0)
        ]

        for label in list(self.order_):
            candidate_set = independent + [label]
            candidate_probabilities, candidate_result = evaluate(
                candidate_set,
                initial_probabilities=current_probabilities,
                start_position=self.order_.index(label),
            )
            candidate_score = float(candidate_result.score)
            improvement = float(candidate_score - current_score)
            accepted = improvement > 1e-12
            if accepted:
                independent.append(label)
                dependent.remove(label)
                current_probabilities = candidate_probabilities
                current_score = float(candidate_score)
                current_result = candidate_result
            history.append(
                self._selection_history_record(
                    len(history),
                    label,
                    accepted,
                    candidate_result,
                    improvement,
                )
            )

        self.evaluated_configurations_ = int(len(probability_cache))
        self._store_selection_result(current_result)
        return sorted(independent), sorted(dependent), current_score, history

    def fit(self, X, Y):
        """Select IL/DL on validation, freeze it, then optionally refit."""
        self._validate_parameters()
        X_array = _as_dense_float(X)
        Y_array = np.asarray(Y, dtype=np.int32)
        if Y_array.ndim != 2 or Y_array.shape[0] != X_array.shape[0]:
            raise ValueError("Y must have shape (n_samples, n_labels).")
        if Y_array.shape[1] == 0 or not np.all(np.isin(Y_array, (0, 1))):
            raise ValueError("Y must contain at least one binary label column.")
        if X_array.shape[0] < 4:
            raise ValueError("GSI_MLC_PA requires at least four training samples.")

        self.n_labels_ = Y_array.shape[1]
        self.partition_mode_ = canonical_partition_mode(self.partition_mode)
        self.requested_final_order_strategy_ = canonical_final_order_strategy(
            self.final_order
        )
        self.selection_order_ = self._validated_order(self.n_labels_)
        self.order_ = list(self.selection_order_)
        selection_started = perf_counter()
        selection_train, validation = self._split_train_validation(X_array, Y_array)
        if len(selection_train) == 0 or len(validation) == 0:
            raise ValueError("The internal train/validation split is empty.")
        self.selection_train_size_ = int(len(selection_train))
        self.validation_size_ = int(len(validation))

        selection_br = self._make_br_model()
        selection_cc = self._make_cc_model(order=self.selection_order_)
        selection_br.fit(X_array[selection_train], Y_array[selection_train])
        selection_cc.fit(X_array[selection_train], Y_array[selection_train])
        self.selection_calibration_audit_ = {
            "br": getattr(selection_br, "calibration_audit_", None),
            "cc": getattr(selection_cc, "calibration_audit_", None),
        }
        if hasattr(selection_cc, "order_"):
            self.order_ = [int(label) for label in selection_cc.order_]
            self.selection_order_ = list(self.order_)

        validation_direct = self._direct_probabilities(
            selection_br, X_array[validation]
        )
        needs_learned_reference = self.partition_mode_ in (
            "learned",
            "learned_no_correlation_order",
            "random_matched",
        )
        needs_stratified_peeling = self.partition_mode_ == "stratified_peeling"
        needs_cv_stratified_peeling = self.partition_mode_ == "cv_stratified_peeling"
        learned_independent = None
        learned_dependent = None
        learned_score = None
        learned_history = None
        learned_evaluation_count = 0
        self.stratified_peeling_audit_ = None

        if needs_cv_stratified_peeling:
            from ..selection.cv_peeling import (
                CVPeelingConfig,
                CVStratifiedPeelingSelector,
            )

            cv_peeling_cfg = CVPeelingConfig(
                threshold=self.stratified_threshold,
                n_folds=self.cv_folds,
                max_depth=self.max_peeling_depth,
                cost=self.cost,
                metric=self.peeling_metric,
                dl_order_direction=self.dl_order_direction,
                aug_normalization=self.aug_normalization,
                decaying_threshold=self.decaying_threshold,
                threshold_decay_step=self.threshold_decay_step,
                random_state=self.random_state,
            )
            cv_peeling_selector = CVStratifiedPeelingSelector(
                config=cv_peeling_cfg,
                base_estimator_factory=lambda: create_binary_estimator(
                    self.base_learner, random_state=self.random_state
                ),
            )
            cv_peeling_res = cv_peeling_selector.fit_partition(
                X_array[selection_train],
                Y_array[selection_train],
            )
            learned_independent = list(cv_peeling_res.all_independent_labels)
            learned_dependent = list(cv_peeling_res.dependent_residual_labels)
            self.stratified_peeling_audit_ = cv_peeling_res.as_dict()
            self.stratified_execution_order_ = list(cv_peeling_res.final_execution_order)
            learned_evaluation_count = cv_peeling_res.num_stages_executed
        elif needs_stratified_peeling:
            from ..selection.stratified_peeling import (
                StratifiedPeelingConfig,
                StratifiedPeelingSelector,
            )
            from ..selection.complexity_penalty import ComplexityPenaltyConfig

            penalty_cfg = ComplexityPenaltyConfig(
                penalty_lambda=self.complexity_penalty_lambda,
                penalty_beta=self.complexity_penalty_beta,
                enabled=self.use_complexity_penalty,
            )
            peeling_cfg = StratifiedPeelingConfig(
                threshold=self.stratified_threshold,
                max_depth=self.max_peeling_depth,
                dl_order_direction=self.dl_order_direction,
                use_complexity_penalty=self.use_complexity_penalty,
                penalty_config=penalty_cfg,
                aug_normalization=self.aug_normalization,
                decaying_threshold=self.decaying_threshold,
                threshold_decay_step=self.threshold_decay_step,
            )
            peeling_selector = StratifiedPeelingSelector(
                config=peeling_cfg,
                base_estimator_factory=lambda: create_binary_estimator(
                    self.base_learner, random_state=self.random_state
                ),
            )
            peeling_res = peeling_selector.fit_partition(
                X_array[selection_train],
                Y_array[selection_train],
                X_array[validation],
                Y_array[validation],
            )
            learned_independent = list(peeling_res.all_independent_labels)
            learned_dependent = list(peeling_res.dependent_residual_labels)
            self.stratified_peeling_audit_ = peeling_res.as_dict()
            self.stratified_execution_order_ = list(peeling_res.final_execution_order)
            learned_evaluation_count = peeling_res.num_stages_executed
        elif needs_learned_reference:
            (
                learned_independent,
                learned_dependent,
                learned_score,
                learned_history,
            ) = self._select_partition(
                X_array[validation],
                Y_array[validation],
                validation_direct,
                selection_cc,
            )
            learned_evaluation_count = int(self.evaluated_configurations_)

        partition = provide_partition(
            self.partition_mode_,
            self.n_labels_,
            learned_independent_labels=learned_independent,
            fixed_independent_labels=self.fixed_independent_labels,
            random_state=(
                self.random_state
                if self.partition_random_state is None
                else int(self.partition_random_state)
            ),
        )
        self.independent_labels_ = list(partition.independent_labels)
        self.dependent_labels_ = list(partition.dependent_labels)
        self.reference_independent_labels_ = (
            None
            if learned_independent is None
            else [int(label) for label in learned_independent]
        )
        self.reference_dependent_labels_ = (
            None
            if learned_dependent is None
            else [int(label) for label in learned_dependent]
        )
        self.reference_validation_objective_ = (
            None if learned_score is None else float(learned_score)
        )
        self.reference_selection_history_ = (
            [] if learned_history is None else list(learned_history)
        )

        if self.partition_mode_ in ("learned", "learned_no_correlation_order"):
            self.validation_objective_ = float(learned_score)
            self.selection_history_ = list(learned_history)
        else:
            (
                self.validation_objective_,
                self.selection_history_,
            ) = self._score_frozen_partition(
                X_array[validation],
                Y_array[validation],
                validation_direct,
                selection_cc,
                self.independent_labels_,
            )
            if self.partition_mode_ == "random_matched":
                self.evaluated_configurations_ += learned_evaluation_count

        self.learned_reference_evaluated_configurations_ = learned_evaluation_count
        self.partition_audit_ = partition.as_dict()
        self.selection_config_.update({
            "base_learner": base_learner_manifest(self.base_learner),
            "partition": dict(self.partition_audit_),
            "stratified_peeling": self.stratified_peeling_audit_,
            "learned_reference_independent_labels": (
                self.reference_independent_labels_
            ),
            "learned_reference_validation_objective": (
                self.reference_validation_objective_
            ),
            "requested_final_order_strategy": (
                self.requested_final_order_strategy_
            ),
        })
        self.selection_time_seconds_ = float(perf_counter() - selection_started)

        # IL/DL is frozen before correlation is calculated. The configured
        # objective/cost may affect selection, but correlation and outer-test
        # data cannot feed back into it.
        self.label_correlation_ = self._compute_label_correlation(Y_array)
        order_dir = (
            "ascending"
            if self.requested_final_order_strategy_ == "ascending_correlation"
            or (
                self.partition_mode_ == "stratified_peeling"
                and self.dl_order_direction == "ascending"
            )
            else "descending"
        )
        correlation_order = self._correlation_order(
            self.label_correlation_,
            self.independent_labels_,
            self.dependent_labels_,
            direction=order_dir,
        )
        self.correlation_order_ = [int(label) for label in correlation_order]
        if self.order is not None:
            desired_final_order = self._validated_order(self.n_labels_)
            self.final_order_strategy_ = "explicit"
        else:
            effective_strategy = self.requested_final_order_strategy_
            if self.partition_mode_ == "learned_no_correlation_order":
                effective_strategy = "selection"
            self.final_order_strategy_ = effective_strategy
            if effective_strategy in (
                "correlation",
                "ascending_correlation",
                "descending_correlation",
            ):
                desired_final_order = list(self.correlation_order_)
            elif effective_strategy == "selection":
                desired_final_order = list(self.selection_order_)
            else:
                desired_final_order = list(range(self.n_labels_))

        # Fit decision policy on true out-of-sample validation probabilities
        # BEFORE any full-data refit occurs to prevent in-sample data leakage.
        self.decision_policy_ = self._make_decision_policy()
        if hasattr(self.decision_policy_, "fit"):
            val_direct = self._direct_probabilities(
                selection_br, X_array[validation]
            )
            val_probs = self._configured_probabilities(
                X_array[validation],
                val_direct,
                selection_cc,
                self.independent_labels_,
            )
            self.decision_policy_.fit(
                val_probs,
                Y_array[validation],
                cost=self.cost,
                penalty=self.penalty,
            )

        # Final fitting uses no validation score and never sees the outer test
        # fold supplied later to predict().
        if self.refit:
            self.br_model_ = self._make_br_model()
            self.br_model_.fit(X_array, Y_array)
            final_x = X_array
            final_y = Y_array
            self.refit_train_size_ = int(X_array.shape[0])
        else:
            self.br_model_ = selection_br
            final_x = X_array[selection_train]
            final_y = Y_array[selection_train]
            self.refit_train_size_ = self.selection_train_size_

        self.cc_model_ = self._make_cc_model(order=desired_final_order)
        self.cc_model_.fit(final_x, final_y)
        if hasattr(self.cc_model_, "order_"):
            self.order_ = [int(label) for label in self.cc_model_.order_]
        else:
            self.order_ = list(desired_final_order)
        self.dependent_parent_map_ = self._dependent_parent_map(
            self.label_correlation_, self.order_, self.dependent_labels_
        )
        final_calibration_audit = {
            "br": getattr(self.br_model_, "calibration_audit_", None),
            "cc": getattr(self.cc_model_, "calibration_audit_", None),
        }
        if any(value is not None for value in final_calibration_audit.values()):
            self.calibration_audit_ = {
                "selection": dict(self.selection_calibration_audit_),
                "final": final_calibration_audit,
                "outer_test_access": False,
            }
        self.selection_config_.update({
            "effective_final_order_strategy": self.final_order_strategy_,
            "selection_order": [int(label) for label in self.selection_order_],
            "correlation_order": [int(label) for label in self.correlation_order_],
            "final_order": [int(label) for label in self.order_],
            "selection_time_seconds": self.selection_time_seconds_,
        })

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        """Return all probabilities under the frozen IL/DL configuration."""
        check_is_fitted(
            self,
            ("is_fitted_", "br_model_", "cc_model_", "independent_labels_"),
        )
        X_array = _as_dense_float(X)
        direct = self._direct_probabilities(self.br_model_, X_array)
        return self._configured_probabilities(
            X_array, direct, self.cc_model_, self.independent_labels_
        )

    def predict(self, X, cost=None):
        """Return {0, abstain, 1} after one final BOP application."""
        probabilities = self.predict_proba(X)
        eff_cost = self.cost if cost is None else cost
        return self.predict_from_proba(probabilities, cost=eff_cost)

    def predict_from_proba(self, probabilities, cost=None):
        """Apply BOP to one already-computed final probability matrix."""
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != self.n_labels_:
            raise ValueError("probabilities must have shape (n_samples, n_labels).")
        return self._apply_bop(
            np.clip(probabilities, 0.0, 1.0), cost=cost
        )

    def predict_full(self, X):
        """Return complete predictions for conventional benchmark metrics."""
        return self.predict_full_from_proba(self.predict_proba(X))

    def predict_full_from_proba(self, probabilities):
        """Threshold one already-computed final probability matrix at 0.5."""
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != self.n_labels_:
            raise ValueError("probabilities must have shape (n_samples, n_labels).")
        return (probabilities >= 0.5).astype(np.int32)

    def decision_function(self, X):
        probabilities = np.clip(self.predict_proba(X), 1e-7, 1.0 - 1e-7)
        return np.log(probabilities / (1.0 - probabilities))


GSIMLCPAClassifier = GSIMLCPartialAbstentionClassifier
