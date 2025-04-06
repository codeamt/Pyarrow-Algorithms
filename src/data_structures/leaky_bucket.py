import pyarrow as pa
import pyarrow.compute as pc
import time

class LeakyBucket:
    """PyArrow-optimized rate limiter with batch operations.

    This class implements a leaky bucket algorithm for rate limiting, 
    leveraging PyArrow for efficient data handling and vectorized 
    operations. It allows you to control the rate at which events or 
    requests are processed, preventing bursts and ensuring a smoother 
    flow.

    The leaky bucket analogy imagines a bucket with a fixed capacity. 
    Tokens (representing events or requests) are added to the bucket 
    at a constant rate. If the bucket is full, new tokens are discarded. 
    Tokens are removed from the bucket when events are processed.

    This implementation supports both single and batch token consumption, 
    as well as an option to wait until tokens become available.

    Attributes:
        capacity (pyarrow.Scalar): The maximum number of tokens the bucket can hold.
        leak_rate (pyarrow.Scalar): The rate at which tokens are added to the bucket (tokens per second).
        _tokens (pyarrow.Scalar): The current number of tokens in the bucket.
        _last_update (pyarrow.Scalar): The timestamp of the last token update.

    Example:
        >>> bucket = LeakyBucket(capacity=10, leak_rate=2)  # 10 tokens, 2 tokens/sec
        >>> if bucket.consume():  # Try to consume 1 token
        ...     # Process the event
        ... else:
        ...     # Rate limit exceeded, wait or drop the event
        >>> consumed_count = bucket.consume_batch(n=5, max_tokens=3)  # Consume up to 3 batches of 5 tokens

    """
    def __init__(self, capacity: int, leak_rate: float):
        self.capacity = pa.scalar(capacity, pa.uint32())
        self.leak_rate = pa.scalar(leak_rate, pa.float32())
        self._tokens = pa.scalar(0.0, pa.float32())
        self._last_update = pa.scalar(time.time(), pa.timestamp('ns'))
        
    def consume(self, n: int = 1 ) -> bool:
        """Atomically consume n tokens"""
        return self.consume_batch(n, 1) == 1
        
    def consume_batch(self, n: int, max_tokens: int) -> int:
        """Batch consume up to max_tokens*n tokens
        Returns:
        Number of successful consumptions (0 to max_tokens)
        """
        now = pa.scalar(time.time(), pa.timestamp('ns'))
        delta = pc.subtract(now, self._last_update).cast(pa.float32())
        leaked = pc.multiply(delta, self.leak_rate)
        self._tokens = pc.max(
            pc.subtract(
                pc.add(self._tokens, leaked),
                self.capacity.cast(pa.float32())
            ),
            pa.scalar(0.0, pa.float32())
        )

        possible = pc.min(
            pc.floor(
                pc.divide(
                    pc.subtract(self.capacity.cast(pa.float32()), self._tokens)),
                    pa.scalar(max_tokens, pa.int32())
                ).cast(pa.int32()
            )
        )
            
        if possible.as_py() > 0:
            self._tokens = pc.add(self._tokens, n * possible)
            self._last_update = now
            
        return possible.as_py()

    def consume_and_wait(self, n: int = 1, timeout: float = None) -> bool:
        """Wait until tokens are available or timeout
        Args:
        n: Tokens required
        timeout: Maximum wait in seconds
        """
        start = time.time()
        while True:
            if self.consume(n):
                return True
            if timeout and (time.time() - start) > timeout:
                return False
            
            # Adaptive backoff using fill ratio
            fill_ratio = pc.divide(self._tokens, self.capacity).as_py()
            sleep_time = 0.001 + (0.05 * fill_ratio)
            time.sleep(sleep_time)  