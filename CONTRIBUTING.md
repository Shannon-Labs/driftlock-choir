# Contributing to Driftlock

Thank you for your interest in contributing to Driftlock! This document provides guidelines for contributing to our research platform for ultra-precise wireless synchronization.

## 🚀 Getting Started

### Prerequisites
- Python ≥ 3.10
- Git
- Basic understanding of signal processing and wireless communication

### Development Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/driftlock.git
cd driftlock

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest
```

## 📋 Contribution Guidelines

### Code Style
- Follow PEP 8 with 4-space indentation
- Use descriptive variable names (snake_case for functions, UpperCamelCase for classes)
- Include type hints for all function parameters and return values
- Add docstrings for all public functions and classes

### Testing
- All new code must include unit tests
- Use deterministic tests with seeded random number generators
- Aim for >80% code coverage
- Run `pytest` before submitting PRs

### Documentation
- Update relevant documentation files
- Include inline comments for complex algorithms
- Update README.md if adding new features

## 🔬 Research Areas

We welcome contributions in these areas:

### Core Algorithm Development
- **Chronometric Handshake**: Improvements to beat signal processing
- **Consensus Algorithms**: Enhanced distributed synchronization
- **Phase Unwrapping**: Better ambiguity resolution techniques
- **Hardware Modeling**: More realistic RF impairments

### Simulation & Validation
- **Monte Carlo Studies**: Extended performance analysis
- **Network Topologies**: Different graph structures and connectivity
- **Mobility Models**: Time-varying network scenarios
- **Benchmarking**: Comparison with existing methods

### Applications
- **5G/6G Integration**: Cellular network synchronization
- **Quantum Networks**: Quantum state measurement coordination
- **Distributed Radar**: Coherent processing applications
- **Financial Systems**: High-frequency trading applications

## 📝 Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to your fork: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### PR Requirements
- Clear description of changes
- Reference to related issues
- Passing tests
- Updated documentation
- Performance impact assessment (if applicable)

## 🐛 Bug Reports

When reporting bugs, please include:
- Python version and operating system
- Steps to reproduce the issue
- Expected vs actual behavior
- Relevant error messages or logs
- Sample code (if applicable)

## 💡 Feature Requests

For feature requests:
- Clearly describe the proposed feature
- Explain the use case and benefits
- Provide implementation suggestions (if any)
- Consider backward compatibility

## 📊 Performance Contributions

For performance-related contributions:
- Include benchmark results
- Document measurement methodology
- Provide before/after comparisons
- Consider memory usage and computational complexity

## 🔒 Security

If you discover a security vulnerability:
- **DO NOT** open a public issue
- Email security@driftlock.dev with details
- Include steps to reproduce
- Allow time for response before public disclosure

## 📄 License

By contributing to Driftlock, you agree that your contributions will be licensed under the MIT License. However, note that the underlying Chronometric Interferometry technology is subject to pending patent applications.

## 🏆 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Acknowledged in relevant research publications
- Invited to co-author papers (for significant contributions)

## 📞 Contact

- **General Questions**: hello@driftlock.dev
- **Technical Issues**: dev@driftlock.dev
- **Research Collaboration**: research@driftlock.dev
- **Security**: security@driftlock.dev

## 🙏 Thank You

Your contributions help advance the state of wireless synchronization technology. We appreciate your interest in pushing the boundaries of what's possible with distributed timing systems!
