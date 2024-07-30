# Deep Translator

Deep Neural Machine Translation

## Prerequisites

This project requires Python 3.11 or later.
To install the `translator` Python package, clone this GitHub repository and simply install the package using pip:

```bash
git clone https://github.com/dobbersc/deep-translator
pip install ./deep-translator
```

This installation also contains the experiment scripts and evaluation results.
If you only are interested in the Python package directly (without the experiments), install the `translator`
package directly from GitHub using pip:

```bash
pip install git+https://github.com/dobbersc/deep-translator@master
```

#### Experiments

To run the scripts in the experiments folder, install the package, including the experiments dependencies:

```bash
git clone https://github.com/dobbersc/deep-translator
pip install -e ./deep-translator[experiments]
```

Afterwords, set the `PYTHONPATH` environment variable to the root of the repository.

- Linux:
    ```bash
    export PYTHONPATH="path/to/repository/root"
    ```

- Windows Powershell:
    ```powershell
    $env:PYTHONPATH = "path/to/repository/root";
    ```

Some of our experiment scripts rely on our trained embeddings and models.
Therefore, train the embeddings and models yourself using the `experiments/train_translator.py` and `experiments/train_translator.py` scripts or download our pretrained resources from our [huggingface](https://huggingface.co/dobbersc/deep-translator) repository.
For the latter choice, move the contents of the [huggingface](https://huggingface.co/dobbersc/deep-translator) repository to the `results` directory.

#### Development

For development, install the package, including the development dependencies:

```bash
git clone https://github.com/dobbersc/deep-translator
pip install -e ./news-deep-translator[dev]
```
