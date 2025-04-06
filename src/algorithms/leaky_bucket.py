import pyarrow as pa
import pyarrow.compute as pc
import time

class LeakyBucket:
    """PyArrow-optimized rate limiter with batch operations
    Args:
    capacity: Maximum token capacity
    leak_rate: Tokens per second
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