# PyArrow Algorithms Toolkit
[![CI Status](https://github.com/codeamt/pyarrow-algorithms/actions/workflows/ci.yml/badge.svg)](https://github.com/codeamt/pyarrow-algorithms/actions)[![Code Coverage](https://codecov.io/gh/codeamt/pyarrow-algorithms/branch/main/graph/badge.svg)](https://codecov.io/gh/codeamt/pyarrow-algorithms)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance foundational system design algorithm implementations using PyArrow and modern Python.



## Features

**Distributed Systems Algorithms**:   
- Consistent Hashing  
- Merkle Trees for synchronization
- Raft Consensus Protocol (TODO) 

**Data Structures**:  
- Bloom Filters  
- HyperLogLog  
- QuadTrees  
- Leaky Bucket rate limiter

**Efficient Computation**:  
- Rsync Algorithm  
- Ray Casting  
- Operational Transformation

## Installation

```bash
# Create virtual environment
python -m venv venvsource venv/bin/activate
# Install with PyArrow
pip install pyarrow==8.0.0 -r requirements.txt
```

## Usage
```python
from pyarrow_algorithms import BloomFilter
bf = BloomFilter(capacity=100000, error_rate=0.01)bf.add("important_item")print("item exists:", "important_item" in bf)
```

## Testing
Run the full test suite with property-based testing:

```bash
pytest tests/ --hypothesis-show-statistics --cov=src
```

## TODOs:
- [ ] Implement Raft Consensus Protocol
- [ ] Add distributed  implementation of key algorithms 
- [ ] Build out a more robust testing/simulation suite with hypothesis and Redis 
- [ ] Add Github Workflows for CI/CD
- [ ] Profile memory usage and develop benchmarks against Vanilla Python/Numpy implementations 

## Contributing
1. Fork the repository
2. Create feature branch 
3. Add tests for new algorithms
4. Submit Pull Request

## License
MIT License - See [LICENSE](LICENSE) for details.
