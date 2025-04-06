import math
import pyarrow as pa
import pyarrow.compute as pc

class LossyCounter:
    """Stream frequency estimator with PyArrow optimizations
    Args:
    epsilon: Maximum error threshold (0 < ε < 1)
    delta: Confidence parameter (0 < δ < 1)
    """
    def __init__(self, epsilon: float, delta: float = 0.01):
        self.epsilon = epsilon
        self.delta = delta
        self.bucket_width = math.ceil(1/epsilon)
        self.current_count = 0
        self.bucket = pa.StructArray.from_arrays(
            [pa.array([], pa.string()), pa.array([], pa.int64())],
            names=['items', 'counts']
        )
        self.error = pa.array([], pa.int64())
        
    def add(self, item: pa.StringScalar):
        """Add an item to the frequency counter"""
        self.current_count += 1
        current_bucket = self.current_count // self.bucket_width
        # Arrow-optimized update
        mask = pc.equal(self.bucket.field('items'), item)
        if pc.any(mask).as_py():
            idx = pc.index(mask).as_py()
            counts = self.bucket.field('counts').set(idx, self.bucket.field('counts')[idx] + 1)
        else:
            self.bucket = self.bucket.append(
                pa.struct([('items', item), ('counts', 1 + current_bucket * self.epsilon)])
            )
            
            self.error = self.error.append(pa.scalar(current_bucket - 1))
        
        if self.current_count % self.bucket_width == 0:
                self.prune()

    def prune(self):
        """Remove infrequent items using Arrow compute"""
        current_bucket = self.current_count // self.bucket_width
        cutoff = current_bucket * self.epsilon
        mask = pc.greater_equal(
            pc.subtract(self.bucket.field('counts'), self.error),
            cutoff
        )
        self.bucket = self.bucket.filter(mask)
        self.error = self.error.filter(mask)

    def get_most_frequent(self, min_support: float = None) -> pa.Table:
        """Get frequent items sorted by estimated frequency
        Args:
        min_support: Minimum support threshold None uses ε)
        Returns:
        Arrow Table with columns: item, count, error, frequency
        """
        min_support = min_support or self.epsilon
        threshold = min_support * self.current_count

        return pa.Table.from_struct(
            self.bucket.filter(
                pc.greater_equal(
                    pc.subtract(self.bucket.field('counts'), self.error),
                    threshold
                )
            )
        ).sort_by([("counts", "descending")]).append_column(
            "frequency",
            pc.subtract(self.bucket.field('counts'), self.error)
        )