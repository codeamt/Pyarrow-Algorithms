import hashlib
import pyarrow as pa

class MerkleTree:
    """
    Data structure for efficient and secure data integrity verification.

    A Merkle tree is a tree-like structure where each leaf node represents 
    a data block, and each non-leaf node represents the cryptographic hash 
    of its child nodes. It allows for efficient verification of data 
    integrity by comparing the root hash with a known value.

    This implementation uses PyArrow for efficient data handling and 
    provides methods to generate and verify Merkle inclusion proofs.

    Attributes:
        leaves (list): A list of data blocks (bytes) representing the leaf nodes 
                       of the Merkle tree.

    Example:
        >>> data_blocks = [b'data1', b'data2', b'data3', b'data4']
        >>> tree = MerkleTree(data_blocks)  # Tree is built during initialization
        >>> proof = tree.get_proof(2)  # Generate proof for the third data block
        >>> is_valid = tree.verify(b'data3', proof)  # Verify the proof
                                                      # (root hash is accessed internally)

    """
    def __init__(self, leaves: list[bytes]):
        """Initialize the MerkleTree with a list of data blocks."""
        self.leaves = leaves
        self.root = self._build_tree()
      
    def _build_tree(self) -> bytes:
      """Build the Merkle tree and return the root hash."""
      current_level = pa.array(self.leaves)
      while len(current_level) > 1:
          # Handle odd number of nodes by duplicating the last node
          if len(current_level) % 2 != 0:
              current_level = current_level.append(current_level[-1])

          # Hash pairs of nodes to create the next level
          current_level = pa.array([
              self._hash(current_level[i] + current_level[i + 1])
              for i in range(0, len(current_level), 2)
          ], type=pa.binary())  # Specify type as binary for byte strings

      # The final remaining node is the root hash
      return current_level[0].as_py()  # Convert to Python bytes
                                
    def _hash(self, data: bytes) -> bytes:
        """Calculate the SHA-256 hash of the given data."""
        return hashlib.sha256(data).digest()

    def get_proof(self, index: int) -> pa.ListArray:
        """
        Generate a Merkle inclusion proof for a leaf node
        Args:
        index: Leaf node index 0-based)
        Returns:
        Arrow ListArray of sibling hashes needed to reconstruct root
        Format: [(is_left: bool, hash: bytes), ...]
        """
        if index < 0 or index >= len(self.leaves):
            raise ValueError("Index out of range")
        
        proof = []
        current_idx = index
        current_level = pa.array(self.leaves)
        
        while len(current_level) > 1:
            # Determine sibling position and hash
            is_left = current_idx % 2
            sibling_idx = current_idx - 1 if is_left else current_idx + 1
            
            # Handle odd-length levels
            if sibling_idx >= len(current_level):
                sibling_hash = current_level[current_idx]
            else:
                sibling_hash = current_level[sibling_idx]
            # Store proof element with positional flag
            proof.append(pa.struct([
                ('is_left', pa.scalar(not is_left)),
                ('hash', self._hash(sibling_hash))
            ]))

            # Move up to parent level
            current_idx = current_idx // 2
            current_level = pa.array([
                self._hash(current_level[i] + current_level[i+1])
                for i in range(0, len(current_level), 2)
            ], type=pa.binary())
        return pa.ListArray.from_arrays(proof)

    def verify(self, leaf: bytes, proof: pa.ListArray) -> bool:
        """Verify a Merkle proof against the known root hash."""
        current_hash = self._hash(leaf)  # Using the private _hash method
        for node in proof:
            if node['is_left'].as_py():
                current_hash = self._hash(node['hash'].as_py() + current_hash)  # Using _hash
            else:
                current_hash = self._hash(current_hash + node['hash'].as_py())  # Using _hash
        return current_hash == self.root  # Comparing with self.root