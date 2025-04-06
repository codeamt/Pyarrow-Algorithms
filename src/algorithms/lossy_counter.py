import math
import pyarrow as pa
import pyarrow.compute as pc

class LossyCounter:
    """Stream frequency estimator with PyArrow optimizations.

    This class implements a Lossy Counting algorithm, a probabilistic data 
    structure used for estimating the frequencies of items in a data stream. 
    It leverages PyArrow for efficient data storage and computation.

    Lossy Counting provides a memory-efficient way to approximate the 
    frequencies of frequent items in large datasets, sacrificing some 
    accuracy for reduced memory usage. It is particularly useful for 
    applications where the exact frequency of every item is not crucial, 
    but identifying the most frequent items is important.

    The algorithm maintains a set of buckets, each containing a subset of 
    items and their estimated counts. Items with lower frequencies are 
    more likely to be discarded, leading to the "lossy" nature of the 
    algorithm. The error in frequency estimates is bounded by the 
    `epsilon` parameter.

    Attributes:
        epsilon (float): The maximum error threshold (0 < ε < 1).
        delta (float): The confidence parameter (0 < δ < 1).
        bucket_width (int): The width of each bucket, calculated based on epsilon.
        current_count (int): The total number of items processed so far.
        bucket (pyarrow.StructArray): A PyArrow StructArray storing the items 
                                     and their estimated counts.
        error (pyarrow.Array): A PyArrow array storing the error values for 
                               each bucket.

    Example:
        >>> counter = LossyCounter(epsilon=0.01)
        >>> counter.add("item1")
        >>> counter.add("item2")
        >>> counter.add("item1")
        >>> frequent_items = counter.get_most_frequent(min_support=0.05)

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