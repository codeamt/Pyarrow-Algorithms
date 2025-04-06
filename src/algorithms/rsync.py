import pyarrow as pa
import pyarrow.compute as pc
import hashlib

class ArrowRsync:
    """PyArrow-optimized Rsync implementation with rolling hash
    Args:
    block_size: Size of data blocks in bytes (default=1024
    window_size: Rolling hash window size (default=64
    """
    def __init__(self, block_size: int = 1024, window_size: int = 64):
        self.block_size = block_size
        self.window_size = window_size
        self.base = 257 # Prime base for rolling hash
        self.mod = 10**18 + 3 # Large prime modulus
        self.base_pow = pow(self.base, window_size, self.mod)
        
    def generate_signature(self, data: pa.Buffer) -> pa.Table:
        """Generate file signature with rolling+strong hashes"""
        blocks = self._chunk_buffer(data)
        rolling_hashes = []
        strong_hashes = []
        for block in blocks:
            rh = self._compute_rolling_hash(block)
            sh = hashlib.md5(block).digest()
            rolling_hashes.append(rh)
            strong_hashes.append(sh)

        return pa.table({
            'block_idx': pa.arange(len(rolling_hashes)),
            'rolling_hash': pa.array(rolling_hashes, pa.uint64()),
            'strong_hash': pa.array(strong_hashes, pa.binary(16))
        })

    def compute_delta(self, source: pa.Buffer, signature: pa.Table) -> pa.Table:
        """Compute delta between source and target signature"""
        source_blocks = self._chunk_buffer(source)
        delta = []
        sig_map = self._create_signature_map(signature)
        for idx, block in enumerate(source_blocks):
            rh = self._compute_rolling_hash(block)
            matches = sig_map.get(rh, pa.array([], pa.int64()))
            if len(matches) > 0:
                block_hash = hashlib.md5(block).digest()
                matches = signature.filter(pc.equal(signature['strong_hash'], block_hash))
                if len(matches) > 0:
                    delta.append(('copy', matches['block_idx'][0].as_py()))
                    continue
            delta.append(('data', block.to_pybytes()))
            
        return pa.Table.from_pydict({
            'operation': [op for op, _ in delta],
            'payload': [val for _, val in delta]
        })

    def apply_patch(self, target: pa.Buffer, delta: pa.Table) -> pa.Buffer:
        """Apply delta to reconstruct source from target"""
        writer = pa.BufferOutputStream()
        for op, payload in zip(delta['operation'], delta['payload']):
            if op == 'copy':
                start = payload * self.block_size
                end = start + self.block_size
                writer.write(target.slice(start, end-start))
            else:
                writer.write(payload)
        return writer.getvalue()

    def _chunk_buffer(self, data: pa.Buffer) -> list:
        """Split buffer into blocks with sliding window"""
        return [data.slice(i, self.block_size)
                for i in range(0, len(data), self.block_size)]

    def _compute_rolling_hash(self, block: pa.Buffer) -> int:
        """Compute polynomial rolling hash using Arrow compute"""
        if len(block) < self.window_size:
            return self._naive_hash(block)
        window = block.slice(0, self.window_size)
        nums = pc.cast(window, pa.uint8()).to_numpy()
        power = self.base ** (self.window_size - 1)
        current_hash = sum(num * (self.base ** (self.window_size - 1 - i))
                           for i, num in enumerate(nums)) % self.mod

        max_hash = current_hash
        for i in range(1, len(block) - self.window_size + 1):
            window = block.slice(i, self.window_size)
            outgoing = nums[i - 1]
            incoming = pc.cast(window[-1:], pa.uint8()).to_numpy()[0]

            current_hash = (current_hash - outgoing * power) * self.base
            current_hash = (current_hash + incoming) % self.mod
            nums = pc.cast(window, pa.uint8()).to_numpy()
            if current_hash > max_hash:
                max_hash = current_hash

            return max_hash

    def _create_signature_map(self, signature: pa.Table) -> dict:
        """Create hash map for quick lookups"""
        return pa.group_by(signature, 'rolling_hash').aggregate([
            ('block_idx', 'list')
        ]).to_pydict()

    def _naive_hash(self, block: pa.Buffer) -> int:
        """Fallback hash for small blocks"""
        return sum(
            byte * (self.base ** (len(block) - i - 1))
            for i, byte in enumerate(block.to_pybytes())
         ) % self.mod
                
        