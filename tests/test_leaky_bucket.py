import pytest
from multiprocessing.pool import ThreadPool
import numpy as np
from hypothesis import given, strategies as st

from data_structures.leaky_bucket import LeakyBucket


class TestLeakyBucket:
    @given(
        rate=st.floats(0.1, 1000),
        capacity=st.integers(1, 1000),
        ops=st.lists(st.integers(1, 5))
    )
    def test_rate_limiting(self, rate, capacity, ops):
        bucket = LeakyBucket(capacity, rate)
        passed = 0
        
        for tokens in ops:
            if bucket.consume(tokens):
                passed += tokens
        
        # Verify never exceeds capacity + rate*time
        max_allowed = capacity + rate * sum(ops) * 1e-3  # Time in ms
        assert passed <= max_allowed * 1.1  # 10% tolerance


    @given(st.data())
    def test_fairness(self, data):
        bucket = LeakyBucket(100, 10)
        threads = data.draw(st.integers(2, 8))
        
        # Simulate concurrent access
        results = ThreadPool(threads).map(
            lambda _: bucket.consume(1), 
            range(100)
        )
        
        # Should allow ~10 ops/s across threads
        assert 8 <= sum(results) <= 12  # ±20% tolerance

# --------------------------
# Running Tests
# --------------------------
#if __name__ == "__main__":
#    pytest.main([
#        "-v", 
#        "--hypothesis-show-statistics",
#        "--cov=leaky-bucket",
#        "--cov-report=html:coverage"
#    ])
# --------------------------