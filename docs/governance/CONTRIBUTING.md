# Contributing to Driftlock Choir

Thank you for your interest in contributing to **Driftlock Choir**! This document provides guidelines for contributing to this open-source framework for ultra-precise distributed timing.

## 🌟 Ways to Contribute

We welcome various types of contributions:

### 🔬 Research Contributions
- **New Algorithms**: Implement novel consensus, estimation, or synchronization algorithms
- **Experimental Validation**: Design and execute new experiments (E14+)
- **Theoretical Analysis**: Add mathematical proofs, bounds analysis, or performance models
- **Hardware Integration**: Bridge simulation to real-world hardware

### 💻 Code Contributions
- **Bug Fixes**: Identify and fix issues in existing code
- **Performance Optimization**: Improve algorithm efficiency or computational speed
- **New Features**: Add functionality (e.g., new oscillator models, channel effects)
- **Test Coverage**: Expand test suites and edge case validation

### 📝 Documentation Contributions
- **Tutorials**: Write guides for specific use cases
- **API Documentation**: Improve docstrings and reference materials
- **Examples**: Create demonstration scripts for new features
- **Translations**: Help make documentation accessible in other languages

### 🎨 Visualization & UX
- **Plotting Tools**: Enhance result visualization
- **Interactive Demos**: Create Jupyter notebooks or web-based demos
- **Audio Representations**: Develop new audible demonstrations of RF concepts

---

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/driftlock-choir.git
cd driftlock-choir/driftlockchoir-oss
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install black isort pylint mypy pytest-cov
```

### 3. Create a Branch

```bash
# Create a descriptive branch name
git checkout -b feature/your-feature-name
# OR
git checkout -b fix/issue-number-description
```

---

## 📋 Development Workflow

### Code Quality Standards

We maintain high code quality through:

#### **1. Code Formatting**
```bash
# Format code with black
black src/ tests/ examples/

# Sort imports with isort
isort src/ tests/ examples/
```

#### **2. Type Checking**
```bash
# Run mypy for type validation
mypy src/ --strict
```

#### **3. Linting**
```bash
# Run pylint for code quality
pylint src/ tests/

# Run flake8 for style compliance
flake8 src/ tests/ --max-line-length=100
```

#### **4. Testing**
```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Coverage must be ≥90% for new code
```

### Commit Message Guidelines

Follow the **Conventional Commits** specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Examples**:
```
feat(consensus): Add hierarchical consensus for large-scale networks

Implements O(log N) hierarchical consensus algorithm with cluster-based
synchronization. Validated on networks up to 1000 nodes.

Closes #42
```

```
fix(estimator): Correct phase unwrapping in beat-note analysis

Phase discontinuities at ±π caused incorrect τ estimation. Added
unwrapping logic with validation tests.

Fixes #73
```

### Pull Request Process

1. **Update Documentation**: Ensure all new features are documented
2. **Add Tests**: Achieve ≥90% coverage for new code
3. **Update CHANGELOG.md**: Document changes under "Unreleased" section
4. **Run Full Test Suite**: `pytest tests/ -v`
5. **Update README**: If adding major features, update relevant sections

**PR Template**:
```markdown
## Description
[Brief description of changes]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation

## Related Issues
Closes #[issue number]
```

---

## 🔬 Research Contribution Guidelines

### Adding New Experiments

When proposing Experiment E14+:

1. **Create Experiment Specification**:
   - **Objective**: What physics/algorithm/scenario does this validate?
   - **Hypothesis**: What do you expect to observe?
   - **Metrics**: How will you measure success?
   - **Validation**: What confirms the results?

2. **Implementation Structure**:
```python
# src/experiments/experiment_eX.py
from src.experiments.base import BaseExperiment

class ExperimentEX(BaseExperiment):
    def create_default_config(self) -> ExperimentConfig:
        # Define parameters
        pass

    def run_experiment(self, context, params):
        # Implement experiment logic
        pass

    def validate_results(self, result):
        # Validation criteria
        pass
```

3. **Add Comprehensive Tests**:
```python
# tests/test_experiment_eX.py
def test_experiment_eX_basic():
    # Test basic functionality
    pass

def test_experiment_eX_edge_cases():
    # Test boundary conditions
    pass

def test_experiment_eX_validation():
    # Test against theoretical predictions
    pass
```

4. **Document Results**:
   - Add summary to `CHANGELOG.md`
   - Create visualization of key findings
   - Update main README if significant

### Adding New Algorithms

When implementing new consensus/estimation algorithms:

1. **Inherit from Base Classes**:
```python
from src.algorithms.base import ConsensusAlgorithm

class YourNewAlgorithm(ConsensusAlgorithm):
    def step(self, node_id, neighbors, measurements):
        # Implementation
        pass
```

2. **Provide Theoretical Justification**:
   - Convergence proof or reference
   - Computational complexity analysis
   - Comparison to existing methods

3. **Benchmark Performance**:
   - Compare against baseline (Metropolis)
   - Test across various topologies
   - Measure convergence speed, accuracy, robustness

---

## 🐛 Reporting Issues

### Bug Reports

Use the **Bug Report** template:
```markdown
**Description**: [Clear description of the bug]

**Steps to Reproduce**:
1. [First step]
2. [Second step]
3. [...]

**Expected Behavior**: [What should happen]

**Actual Behavior**: [What actually happens]

**Environment**:
- OS: [e.g., macOS 14.5]
- Python version: [e.g., 3.10.2]
- Package versions: [output of `pip list`]

**Additional Context**: [Screenshots, logs, etc.]
```

### Feature Requests

Use the **Feature Request** template:
```markdown
**Problem**: [What problem does this solve?]

**Proposed Solution**: [Your idea]

**Alternatives Considered**: [Other approaches]

**Use Cases**: [Who would benefit and how?]
```

---

## 📚 Documentation Style Guide

### Docstring Format

Use **Google-style** docstrings:

```python
def estimate_tau_delta_f(signal: np.ndarray,
                         sample_rate: float,
                         carrier_freq: float) -> Tuple[float, float]:
    """Estimate time-of-flight and frequency offset from beat-note signal.

    Applies phase-slope analysis to extract τ (time-of-flight) and Δf
    (frequency offset) from heterodyne mixing of two oscillators.

    Args:
        signal: Complex-valued beat-note signal (I+jQ)
        sample_rate: Sampling frequency in Hz
        carrier_freq: Nominal carrier frequency in Hz

    Returns:
        Tuple of (tau_ps, delta_f_hz) where:
            - tau_ps: Time-of-flight in picoseconds
            - delta_f_hz: Frequency offset in Hz

    Raises:
        ValueError: If signal length < 1000 samples

    Example:
        >>> signal = generate_beat_note(tau_ps=10.0, delta_f=100.0)
        >>> tau_est, df_est = estimate_tau_delta_f(signal, 1e6, 2.4e9)
        >>> print(f"τ = {tau_est:.2f} ps, Δf = {df_est:.2f} Hz")
        τ = 10.05 ps, Δf = 100.02 Hz
    """
    # Implementation
    pass
```

### README Updates

- **Be Concise**: Avoid overly technical jargon in main README
- **Use Examples**: Show, don't just tell
- **Add Visuals**: Plots, diagrams, or audio when helpful
- **Link to Details**: Use separate docs for deep dives

---

## 🎯 Code Review Checklist

Reviewers will check:

- [ ] **Correctness**: Does the code do what it claims?
- [ ] **Tests**: Are there comprehensive tests? Do they pass?
- [ ] **Documentation**: Are docstrings complete and accurate?
- [ ] **Performance**: Are there obvious optimization opportunities?
- [ ] **Style**: Does it follow project conventions?
- [ ] **Backward Compatibility**: Does it break existing APIs?

---

## 🌍 Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- **Be Respectful**: Treat all contributors with respect
- **Be Collaborative**: Value diverse perspectives and approaches
- **Be Professional**: Focus on technical merit, not personal attacks
- **Be Open-Minded**: Consider alternative solutions and feedback

See **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)** for our full community guidelines.

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, brainstorming, research ideas
- **Email** (hunter@shannonlabs.dev): Private or sensitive matters

---

## 🏆 Recognition

We value all contributions! Contributors will be:

- Listed in **CONTRIBUTORS.md**
- Acknowledged in release notes for significant contributions
- Invited to co-author academic papers if contributing novel algorithms/experiments

---

## 📝 License

By contributing to Driftlock Choir, you agree that your contributions will be licensed under the **MIT License**.

---

## ❓ Questions?

- **Not sure where to start?** Check [Good First Issues](https://github.com/Shannon-Labs/driftlock-choir/labels/good%20first%20issue)
- **Have questions?** Open a [Discussion](https://github.com/Shannon-Labs/driftlock-choir/discussions)
- **Want to propose something major?** Email hunter@shannonlabs.dev to discuss

---

Thank you for helping advance ultra-precise distributed timing! 🎵⚡
