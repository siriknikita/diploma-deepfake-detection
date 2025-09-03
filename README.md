# DeepFake Detection

DeepFake Detection is an open-source project designed to detect and analyze manipulated media using machine learning and computer vision.
It provides a modular structure, easy configuration management with Pydantic and YAML, and utilities for preprocessing, training, and evaluating deepfake detection models.

---

## Getting Started

### Prerequisites

* **Python**: >= 3.9
* **pip**: >= 21.0
* **virtualenv** or **conda** recommended
* **System packages** (Linux/macOS): will be provided later
* **Git** for version control

---

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/siriknikita/diploma-deepfake-detection.git
   cd diploma-deepfake-detection
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Usage

### Configuration

All configuration files live under `configs/`.
You can override defaults by editing YAML files or defining new ones.
Pydantic models ensure validation and type safety.

Example: `configs/default.yaml`

```yaml
window_size: 9
padding: 0.25
```

---

### Running the Project

**Run preprocessing**

```bash
will be provided later
```

**Run training**

```bash
will be provided later
```

**Run evaluation**

```bash
will be provided later
```

---

## Development

### Code Quality Checks

Run the following commands manually:

* **Linting check**:

  ```bash
  ./scripts/check-linting.sh
  ```
* **Formatting check**:

  ```bash
  ./scripts/check-formatting.sh
  ```
* **Run formatting**:

  ```bash
  ./scripts/format-src-files.sh
  ```
* **Type checking**:

  ```bash
  ./scripts/check-static-types.sh
  ```

---

## Contributing

We welcome contributions! 🎉

Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on pull requests, coding standards, and commit message format.

---

## Security

If you discover a security vulnerability, please review our [SECURITY.md](SECURITY.md) for supported versions and reporting process.

---

## License

This project is licensed under the **MIT License**.
See [LICENSE](LICENSE) for more information.

---

## Acknowledgements

* [Pydantic](https://docs.pydantic.dev/) for config validation
* [OpenCV](https://opencv.org/) for computer vision operations
* [PyTorch](https://pytorch.org/) for deep learning models
* The open-source community for inspiration and tools
