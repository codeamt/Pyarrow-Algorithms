import pyarrow as pa
import pyarrow.compute as pc
import hashlib

class ConsistentHash:
    """PyArrow-optimized consistent hashing with weighted nodes.

    This class implements a consistent hashing algorithm using PyArrow for 
    efficient storage and operations. It supports weighted nodes, allowing 
    for uneven distribution of data across nodes based on their capacity 
    or other factors.

    Consistent hashing ensures that when a node is added or removed, 
    only a minimal number of keys need to be remapped to different nodes, 
    minimizing data movement and disruption.

    Attributes:
        replicas (int): The base number of virtual nodes per weight unit.
        ring (pyarrow.Table): A PyArrow Table storing the hash values and 
                              corresponding nodes, representing the hash ring.

    Example:
        >>> nodes = {"node1": 1, "node2": 2}  # Node weights
        >>> ch = ConsistentHash(nodes)
        >>> ch.get_node("key1")  # Returns the node responsible for "key1"
        'node2'
        >>> ch.add_weighted_node("node3", weight=3)
        >>> ch.remove_node("node1")

    """
    def __init__(self, nodes: dict, replicas: int = 100):
        self.replicas = replicas
        self.ring = pa.Table.from_arrays(
            arrays=[[], []],
            names=['hash', 'node'],
            schema=pa.schema([
                ('hash', pa.uint64()),
                ('node', pa.string())
            ])
        )
        for node, weight in nodes.items():
            self.add_weighted_node(node, weight)

    def add_weighted_node(self, node: str, weight: int = 1):
        """Add node with weight using vectorized operations"""
        virtual_nodes = weight * self.replicas
        hashes = pa.array([
            int.from_bytes(
                hashlib.blake2s(f"{node}-{i}".encode()).digest()[:8],
                'little'
            ) for i in range(virtual_nodes)
        ], type=pa.uint64())
        nodes = pa.array([node] * virtual_nodes)
        new_entries = pa.Table.from_arrays(
        [hashes, nodes],
        names=['hash', 'node']
        )
        self.ring = pa.concat_tables([self.ring, new_entries])
        self.ring = self.ring.sort_by('hash')

    def remove_node(self, node: str):
        """Remove all virtual nodes for a physical node"""
        mask = pc.not_equal(self.ring['node'], node)
        self.ring = self.ring.filter(mask)

    def get_node(self, key: str) -> str:
        """Find node for key using binary search"""
        key_hash = int.from_bytes(
            hashlib.blake2s(key.encode()).digest()[:8],
            'little'
        )
        hashes = self.ring['hash'].combine_chunks()
        idx = pc.binary_search(hashes, value=key_hash)
        if idx == len(hashes):
            idx = 0
        return self.ring['node'][idx].as_py()

    def balance_quality(self) -> float:
        """Calculate balance quality (0-1) using Arrow compute"""
        total_vnodes = len(self.ring)
        node_counts = pc.value_counts(self.ring['node'])
        counts = node_counts['counts'].combine_chunks()
        return pc.stddev(counts).as_py() / (total_vnodes / len(counts))