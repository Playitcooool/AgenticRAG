from __future__ import annotations

import unittest

from agentic_rag.vector_backends import TurboVecIndex, choose_pq_subquantizers, faiss_pq_min_training_points


class FaissPQConfigTest(unittest.TestCase):
    def test_min_training_points_match_faiss_recommendation(self) -> None:
        self.assertEqual(156, faiss_pq_min_training_points(2))
        self.assertEqual(624, faiss_pq_min_training_points(4))

    def test_choose_subquantizers_divides_dimension(self) -> None:
        self.assertEqual(96, choose_pq_subquantizers(768))
        self.assertEqual(96, choose_pq_subquantizers(384))
        self.assertEqual(1, choose_pq_subquantizers(7, max_subquantizers=4))


class TurboVecIndexTest(unittest.TestCase):
    def test_search_uses_2d_query_matrix_and_first_result_row(self) -> None:
        class FakeNumpy:
            def asarray(self, value, dtype=None):
                return value

        class FakeTurboIndex:
            def __init__(self) -> None:
                self.query = None

            def search(self, query, k):
                self.query = query
                return [[0.9, 0.7]], [[3, 5]]

        index = object.__new__(TurboVecIndex)
        index._np = FakeNumpy()
        index._index = FakeTurboIndex()

        self.assertEqual([(3, 0.9), (5, 0.7)], index.search([1.0, 2.0], k=2))
        self.assertEqual([[1.0, 2.0]], index._index.query)


if __name__ == "__main__":
    unittest.main()
