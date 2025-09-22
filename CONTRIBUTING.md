# Contributing to Helix

Thank you for your interest in contributing to Helix! We welcome contributions from the community and are excited to work with you.

## 🚀 Getting Started

### Prerequisites

- Rust 1.70+
- Git
- For Go bindings: Go 1.19+

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Rx947getrexp/Helix.git
   cd helix
   ```

2. **Build the project**
   ```bash
   cargo build
   ```

3. **Run tests**
   ```bash
   cargo test
   ```

4. **Run benchmarks** (optional)
   ```bash
   cargo bench
   ```

## 📋 Development Guidelines

### Code Standards

Please follow the guidelines in [docs/development/COMMON_CODE_GUIDE.md](docs/development/COMMON_CODE_GUIDE.md):

- **Unified error handling**: Use `HelixResult<T>` and `HelixError`
- **Consistent naming**: Follow Rust naming conventions
- **Documentation**: Document all public APIs
- **Testing**: Write comprehensive tests for new features
- **Performance**: Consider performance implications

### Before Making Changes

1. **Read the documentation**:
   - [docs/development/COMMON_CODE_GUIDE.md](docs/development/COMMON_CODE_GUIDE.md)
   - [docs/development/IMPLEMENTATION_PLAN.md](docs/development/IMPLEMENTATION_PLAN.md)

2. **Check existing issues** to avoid duplicate work

3. **Create an issue** for significant changes to discuss the approach

### Code Quality

- **Format your code**: `cargo fmt`
- **Check for lints**: `cargo clippy`
- **Run security audit**: `cargo audit`
- **Test coverage**: Aim for >95% coverage

## 🔄 Contribution Process

### 1. Fork and Clone

```bash
git fork https://github.com/Rx947getrexp/Helix.git
git clone https://github.com/YOUR-USERNAME/helix.git
cd helix
git remote add upstream https://github.com/Rx947getrexp/Helix.git
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number
```

### 3. Make Changes

- Follow the [development guidelines](#development-guidelines)
- Write tests for your changes
- Update documentation as needed
- Ensure all tests pass

### 4. Commit Changes

Use clear, descriptive commit messages:

```bash
git commit -m "feat: add vector quantization support

- Implement 8-bit and 4-bit quantization
- Add quantization configuration options
- Update storage format to support quantized vectors
- Add comprehensive tests for quantization accuracy"
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request through GitHub.

## 🐛 Reporting Bugs

When reporting bugs, please include:

1. **Environment**:
   - Rust version (`rustc --version`)
   - Operating system
   - Helix version

2. **Steps to reproduce**:
   - Minimal code example
   - Expected behavior
   - Actual behavior

3. **Additional context**:
   - Error messages
   - Stack traces
   - Performance measurements (if relevant)

## 💡 Feature Requests

For feature requests:

1. **Check existing issues** first
2. **Describe the use case** clearly
3. **Explain the benefit** to users
4. **Consider implementation complexity**
5. **Propose an API design** if applicable

## 🏗️ Types of Contributions

### Code Contributions

- **Bug fixes**: Fix issues in existing functionality
- **Features**: Implement new capabilities
- **Performance**: Optimize existing code
- **Tests**: Improve test coverage
- **Documentation**: Improve code documentation

### Non-Code Contributions

- **Documentation**: Improve guides and tutorials
- **Examples**: Add usage examples
- **Issues**: Report bugs and suggest features
- **Reviews**: Review pull requests
- **Translations**: Translate documentation

## 📚 Development Areas

We especially welcome contributions in these areas:

### Core Features
- Vector quantization (4-bit, 8-bit)
- Additional distance metrics
- Auto-scaling index selection
- Query optimization

### Language Bindings
- Python bindings
- Node.js bindings
- WebAssembly support

### Tooling
- CLI enhancements
- Monitoring and observability
- Development tools

### Documentation
- More examples
- Performance guides
- Integration tutorials

## 🧪 Testing

### Running Tests

```bash
# Unit tests
cargo test

# Integration tests
cargo test --test integration_tests

# CLI tests
cargo test --test cli_integration_tests --features cli

# Go bindings tests
cd go && go test
```

### Writing Tests

- **Unit tests**: Test individual functions and modules
- **Integration tests**: Test end-to-end workflows
- **Property tests**: Use `proptest` for randomized testing
- **Benchmarks**: Measure performance regressions

### Test Guidelines

- Test both success and error paths
- Use descriptive test names
- Include edge cases
- Verify thread safety for concurrent code

## 📊 Performance Considerations

When making changes that could affect performance:

1. **Run benchmarks** before and after changes
2. **Profile memory usage** for memory-intensive changes
3. **Consider algorithmic complexity**
4. **Test with realistic data sizes**

```bash
# Run benchmarks
cargo bench

# Profile with different data sizes
cargo run --release --bin benchmark -- --size 100000
```

## 🔧 Debugging

### Common Debug Tools

```bash
# Enable debug logging
RUST_LOG=debug cargo test

# Memory debugging with valgrind
cargo build && valgrind ./target/debug/helix

# Performance profiling
cargo build --release
perf record ./target/release/helix
perf report
```

## 📖 Documentation

### API Documentation

- Document all public functions with examples
- Use `cargo doc` to generate documentation
- Include performance characteristics

### Guides and Tutorials

- Keep examples up-to-date
- Include complete, runnable examples
- Explain the "why" not just the "how"

## 🤝 Code Review Process

### For Contributors

- Keep PRs focused and reasonably sized
- Write clear PR descriptions
- Respond to feedback promptly
- Update PRs based on review comments

### For Reviewers

- Be constructive and respectful
- Focus on code quality and correctness
- Consider performance implications
- Suggest improvements clearly

## 📄 License

By contributing to Helix, you agree that your contributions will be licensed under the same terms as the project (MIT/Apache-2.0 dual license).

## 🙋 Getting Help

- **GitHub Discussions**: For general questions
- **GitHub Issues**: For bugs and feature requests
- **Discord**: [Join our community](https://discord.gg/helix) (if applicable)

## 🏆 Recognition

Contributors are recognized in:

- Release notes for significant contributions
- `CONTRIBUTORS.md` file
- GitHub contributor statistics

Thank you for helping make Helix better! 🚀