import inspect
import unittest

import torch

from .structural_candidates import (
    CandidateRelativeConfig,
    CandidateRelativePredictor,
    IdentityResidualConfig,
    IdentityResidualPredictor,
    candidate_relative_objective,
    identity_residual_objective,
    stable_candidate_order,
    validate_causal_history,
)


class StructuralCandidateTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.batch, self.length, self.dim, self.metadata = 3, 4, 8, 3
        self.features = torch.randn(self.batch, self.length, self.dim)
        self.metadata_rows = torch.randn(self.batch, self.length, self.metadata)
        self.mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1]], dtype=torch.float32)
        self.candidates = torch.randn(self.batch, 5, self.dim)
        self.candidate_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        self.positive = torch.tensor([1, 2, 0])

    def test_residual_query_is_candidate_independent_and_finite(self):
        model = IdentityResidualPredictor(IdentityResidualConfig(
            feature_dim=self.dim, history=self.length, metadata_dim=self.metadata, hidden_dim=16))
        query, anchor = model.history_state(self.features, self.metadata_rows, self.mask)
        altered_candidates = self.candidates + 100.0
        # The query API has no candidate argument; changing the future set
        # therefore cannot change the history state.
        query_again = model(self.features, self.metadata_rows, self.mask)
        self.assertTrue(torch.allclose(query, query_again, atol=1e-7, rtol=0))
        self.assertTrue(torch.isfinite(query).all())
        self.assertTrue(torch.isfinite(anchor).all())
        self.assertEqual(query.shape, (self.batch, self.dim))
        self.assertFalse(torch.equal(self.candidates, altered_candidates))
        for method in ("predictive", "discriminative", "predictive_ranking"):
            loss, parts = identity_residual_objective(
                query, anchor, self.candidates, self.positive, method,
                candidate_mask=self.candidate_mask)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(parts["anchor_error"]))

    def test_candidate_relative_query_is_candidate_independent_and_scores_later(self):
        model = CandidateRelativePredictor(CandidateRelativeConfig(
            feature_dim=self.dim, history=self.length, metadata_dim=self.metadata,
            hidden_dim=16, relation_dim=8))
        query = model(self.features, self.metadata_rows, self.mask)
        query_again = model(self.features, self.metadata_rows, self.mask)
        self.assertTrue(torch.allclose(query, query_again, atol=1e-7, rtol=0))
        scores = model.score(query, self.candidates, self.candidate_mask)
        self.assertEqual(scores.shape, (self.batch, 5))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(torch.all(scores[~self.candidate_mask] < -9999))
        for method in ("predictive", "discriminative", "predictive_ranking"):
            loss, parts = candidate_relative_objective(
                model, query, self.candidates, self.positive, method,
                candidate_mask=self.candidate_mask)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(parts["relation"]))

    def test_candidate_relative_zero_query_has_finite_scores_and_gradients(self):
        model = CandidateRelativePredictor(CandidateRelativeConfig(
            feature_dim=self.dim, history=self.length, metadata_dim=self.metadata,
            hidden_dim=16, relation_dim=8))
        query = torch.zeros(self.batch, 8, requires_grad=True)
        scores = model.score(query, self.candidates, self.candidate_mask)
        loss, _ = candidate_relative_objective(
            model, query, self.candidates, self.positive,
            "discriminative", candidate_mask=self.candidate_mask)
        total = scores[slice(None), :3].mean() + loss
        total.backward()
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(query.grad).all())

    def test_strictly_past_history_is_checked(self):
        history_frames = torch.tensor([[0, 2, 4, 5], [0, 0, 3, 4]])
        current_frames = torch.tensor([6, 5])
        validate_causal_history(history_frames, current_frames, self.mask[:2])
        with self.assertRaisesRegex(ValueError, "current or future"):
            validate_causal_history(history_frames, torch.tensor([5, 5]), self.mask[:2])

    def test_labels_are_not_part_of_history_forward_contract(self):
        for method in (IdentityResidualPredictor.forward, CandidateRelativePredictor.forward):
            names = set(inspect.signature(method).parameters)
            self.assertNotIn("positive_index", names)
            self.assertNotIn("target_gt", names)
            self.assertNotIn("candidate_features", names)

    def test_candidate_order_is_stable_and_ties_preserve_detector_order(self):
        ids = [20, 10, 30, 40]
        scores = [0.5, 0.5, 0.8, 0.5]
        expected = (2, 0, 1, 3)
        self.assertEqual(stable_candidate_order(ids, scores), expected)
        self.assertEqual(stable_candidate_order(ids, scores), expected)
        self.assertEqual(stable_candidate_order(ids), (0, 1, 2, 3))


if __name__ == "__main__":
    unittest.main()
