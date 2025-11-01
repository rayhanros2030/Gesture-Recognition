# Contributing to Gesture Recognition

Thank you for your interest in contributing! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, dependencies)

### Suggesting Features

Feature suggestions are welcome! Please include:
- Description of the feature
- Use case and motivation
- Implementation ideas (if any)

### Code Contributions

#### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Gesture-Recognition.git
cd Gesture-Recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install black flake8 pytest
```

#### Making Changes

1. **Create a branch**: `git checkout -b feature/your-feature-name`
2. **Make changes**: Follow the coding guidelines below
3. **Test**: Ensure all tests pass
4. **Commit**: Write clear commit messages
5. **Push**: `git push origin feature/your-feature-name`
6. **Pull Request**: Open a PR with clear description

#### Coding Guidelines

- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to functions/classes
- Comment complex logic
- Keep functions focused and small
- Write tests for new features

#### Code Style

```python
# Good
def recognize_gesture(landmarks):
    """
    Recognize gesture from landmarks.
    
    Args:
        landmarks: Hand landmarks from MediaPipe
        
    Returns:
        str: Gesture name
    """
    # Implementation
    pass

# Format code with black
black your_file.py
```

#### Testing

```bash
# Run tests
pytest tests/

# Check coverage
pytest --cov=src tests/

# Lint code
flake8 src/
```

### Pull Request Process

1. Update documentation if needed
2. Ensure tests pass
3. Request review from maintainers
4. Address feedback
5. Once approved, maintainer will merge

## 📝 Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Update or create relevant documentation files

## 🎯 Areas for Contribution

### Good First Issues
- Improve finger counting accuracy
- Add more gesture types
- Enhance visualizations
- Better error handling
- Documentation improvements

### Advanced Contributions
- Optimize model performance
- Add new ML architectures
- Implement gesture recording UI
- Cross-platform improvements
- Cloud deployment guides

## ❓ Questions?

Feel free to open an issue or discussion for any questions!

## 📜 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Collaborate with others

Thank you for contributing! 🎉

