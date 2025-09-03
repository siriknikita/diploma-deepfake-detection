# Contributing to DeepFake Detection

We welcome contributions to the **DeepFake Detection** project! This document outlines guidelines for contributing, reporting issues, and submitting code.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How to Contribute](#how-to-contribute)

   * [Reporting Bugs](#reporting-bugs)
   * [Requesting Features](#requesting-features)
   * [Submitting Code](#submitting-code)
3. [Development Setup](#development-setup)
4. [Coding Guidelines](#coding-guidelines)
5. [License](#license)

---

## Code of Conduct

By participating in this project, you agree to follow the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). Please be respectful and constructive in all communications.

---

## How to Contribute

### Reporting Bugs

If you encounter a bug:

1. Check the [issues](https://github.com/siriknikita/diploma-deepfake-detection/issues) to see if it has already been reported.
2. Open a new issue with:

   * A clear title
   * Steps to reproduce
   * Expected and actual behavior
   * Screenshots or logs, if applicable

### Requesting Features

To request a new feature:

1. Open an issue labeled `enhancement`.
2. Describe the feature clearly and explain its value to the project.
3. Include examples or references if relevant.

### Submitting Code

1. Fork the repository and clone it locally.
2. Create a new branch for your changes:

   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and ensure all code follows the guidelines (see below).
4. Run linting, and formatting:

   ```bash
   ./scripts/run-all-checks.sh
   ```
5. Commit changes with a descriptive message:

   ```bash
   git commit -m "Add feature X to preprocessing module"
   ```
6. Push your branch and open a pull request.

**Note:** Pull requests should not be squashed; maintain individual commits if possible.

---

## Development Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/siriknikita/diploma-deepfake-detection.git
   cd DeepFake-Detection
   ```
2. Install dependencies via uv:

   ```bash
   uv sync --dev
   ```
3. Set up your environment (optional):

   ```bash
   uv shell
   ```
4. Run tests or experiments as needed.

---

## Coding Guidelines

* Follow **PEP8** standards and use `black` for formatting.
* Use **type annotations** and Pydantic models where appropriate.
* Use **absolute imports** from `src/`.
* Include unit tests for any new functionality.
* Keep functions modular and reusable.
* Document all public functions and classes with docstrings.

---

## License

By contributing, you agree that your contributions will be licensed under the project’s [MIT License](LICENSE).
