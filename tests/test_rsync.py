import pytest
import hashlib
import pyarrow as pa
from hypothesis import given, strategies as st

from src.algorithms.rsync import ArrowRsync


class TestRsync:
    @given(
        source=st.binary(min_size=1),
        target=st.binary(min_size=1)
    )
    def test_rsync_reconstruction(self, source, target):
        rsync = ArrowRsync()
        target_sig = rsync.generate_signature(pa.py_buffer(target))
        delta = rsync.compute_delta(pa.py_buffer(source), target_sig)
        reconstructed = rsync.apply_patch(pa.py_buffer(target), delta)
        assert reconstructed.to_pybytes() == source
    
    def test_rolling_hash_collision(self):
        # Craft collision
        rsync = ArrowRsync()
        block1 = b"A"*64 + b"XXXX"
        block2 = b"B"*64 + b"XXXX"
        assert rsync._compute_rolling_hash(block1) == \
            rsync._compute_rolling_hash(block2) # But strong hashes differ
        assert hashlib.md5(block1).digest() != \
            hashlib.md5(block2).digest()

# --------------------------
# Running Tests
# --------------------------
#if __name__ == "__main__":
#    pytest.main([
#        "-v", 
#        "--hypothesis-show-statistics",
#        "--cov=rsync",
#        "--cov-report=html:coverage"
#    ])
# --------------------------

