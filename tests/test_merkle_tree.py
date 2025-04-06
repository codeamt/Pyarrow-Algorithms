import pytest
import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
from hypothesis import given, strategies as st

from src.algorithms.merkle_tree import MerkleTree

class TestMerkleTree:
    @given(st.lists(st.binary(), min_size=1))
    def test_merkle_proofs(self, data):
        tree = MerkleTree(data)
        
        for i in range(len(data)):
            proof = tree.get_proof(i)
            assert MerkleTree.verify(tree.root, data[i], proof)

    def test_tamper_detection(self):
        data = [b"a", b"b", b"c"]
        tree1 = MerkleTree(data)
        tree2 = MerkleTree([b"a", b"x", b"c"])
        assert tree1.root != tree2.root

# --------------------------
# Running Tests
# --------------------------
#if __name__ == "__main__":
#    pytest.main([
#        "-v", 
#        "--hypothesis-show-statistics",
#        "--cov=merkle-tree",
#        "--cov-report=html:coverage"
#    ])
# --------------------------