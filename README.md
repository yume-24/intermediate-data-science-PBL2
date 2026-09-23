# Political Lean Classification with NLP

An experimental NLP project that classifies Reddit posts using the dataset's **Liberal** and **Conservative** labels. It investigates whether sentiment, moral-language signals, and contextual embeddings improve on a TF-IDF baseline.

The project combines pretrained transformer feature extraction with classical machine learning, then compares predictive performance, feature contributions, and runtime. In the main saved logistic regression comparison, **TF-IDF alone achieved 78.18% test accuracy and 0.7607 macro F1**, outperforming the combined feature model.

**Stack:** Python · pandas · NumPy · scikit-learn · PyTorch · Hugging Face Transformers · Sentence Transformers · Matplotlib · Jupyter

## Project highlights

- Built three feature representations: TF-IDF, semantic features, and a combination of both.
- Extracted sentiment probabilities and 768-dimensional RoBERTa document embeddings.
- Engineered moral-language similarity scores and virtue–violation contrasts using sentence embeddings.
- Compared logistic regression, linear SVM, SGD linear SVM, RBF SVM, and Extra Trees across feature sets.
- Used stratified train/test splitting, cross-validated hyperparameter search, and macro F1 to account for class imbalance.
- Explored model coefficients, confusion matrices, feature ablations, misclassified examples, and performance–runtime tradeoffs.

## Approach

The main workflow is in [`full_classifier_virtue_violation_patched.ipynb`](full_classifier_virtue_violation_patched.ipynb).

1. **Prepare the text.** Combine each post's title and body, fill missing text, retain the two target labels, and remove empty posts.
2. **Extract features.** Build lexical and semantic representations of the same combined text.
3. **Train and tune.** Fit preprocessing and classifiers inside scikit-learn pipelines, selecting hyperparameters on the training split.
4. **Evaluate and interpret.** Compare held-out predictions and examine which feature groups contribute useful information.

| Feature block | Representation |
| --- | --- |
| TF-IDF | Word n-grams, with up to 50,000 features |
| Sentiment | Negative, neutral, and positive probabilities from `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Contextual embeddings | 768 features from `roberta-base`, using mean pooling over non-padding tokens and L2 normalization |
| Moral-language similarities | 12 cosine similarities between MiniLM post embeddings and averaged prototype-description embeddings |
| Moral contrasts | 6 differences between paired virtue and violation scores |

The moral-language pairs are **care / harm**, **fairness / cheating**, **loyalty / betrayal**, **authority / subversion**, **purity / degradation**, and **liberty / oppression**. Together with sentiment and RoBERTa embeddings, these produce **789 numeric semantic features**. The transformer models are used as pretrained feature extractors; they are not fine-tuned in this workflow.

## Results

The main notebook's saved run contains **12,854 posts**: 8,319 labeled Liberal and 4,535 labeled Conservative. It uses a stratified 80/20 split with `random_state=42`: **10,283 training posts** and **2,571 test posts**.

The following results come from the notebook's main logistic regression comparison. Each model uses the same train/test split, with five-fold grid search on the training set and macro F1 as the selection metric.

| Features | Test accuracy | Test macro F1 | Best CV macro F1 |
| --- | ---: | ---: | ---: |
| TF-IDF only | **78.18%** | **0.7607** | **0.7492** |
| Semantic only | 70.87% | 0.6958 | 0.6928 |
| TF-IDF + semantic | 76.86% | 0.7517 | 0.7416 |

**Finding:** Adding semantic features did not improve the tuned logistic regression baseline in this experiment. This makes the simpler lexical model a useful reference when assessing the added complexity of transformer features.

A separate classifier sweep is saved in [`classifier_results_so_far.csv`](classifier_results_so_far.csv). Its highest recorded test macro F1 is **0.7497**, from a linear SVM with TF-IDF and semantic features. That sweep uses different search settings from the comparison above, so its scores should be read as a separate experiment.

These are recorded notebook and CSV results, not a fresh reproduction. Earlier notebooks retain outputs from different experimental states.

## Repository guide

| File | Purpose |
| --- | --- |
| [`full_classifier_virtue_violation_patched.ipynb`](full_classifier_virtue_violation_patched.ipynb) | Main experiment: feature extraction, model comparisons, interpretation, and semantic ablations |
| [`full_classifier_with_roberta_embeddings_integrated.ipynb`](full_classifier_with_roberta_embeddings_integrated.ipynb) | Earlier integrated workflow with RoBERTa embeddings and six moral-language scores |
| [`full_classifier_with_roberta_embeddings.ipynb`](full_classifier_with_roberta_embeddings.ipynb) | Intermediate RoBERTa embedding experiment |
| [`full_classifier.ipynb`](full_classifier.ipynb) | Earlier TF-IDF, sentiment, and moral-language comparison |
| [`classifier.ipynb`](classifier.ipynb) | Initial classification experiments |
| [`eda.ipynb`](eda.ipynb) | Exploratory data analysis and initial embedding experiments |
| [`classifier_results_so_far.csv`](classifier_results_so_far.csv) | Saved classifier sweep metrics, runtimes, and selected parameters |
| `data/reddit_dr.csv` | Input dataset for the classification workflow |
| `reddit_with_semantic_features.csv` | Generated feature table; recreated by the feature-extraction workflow |

## Running the project

### 1. Set up an environment

From the repository root, create a Python 3 environment and install the notebook dependencies. The commands below use macOS/Linux shell syntax.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyterlab ipykernel pandas numpy scikit-learn torch transformers sentence-transformers matplotlib seaborn tqdm
python -m ipykernel install --user --name pbl2 --display-name "Python (PBL2)"
python -m jupyterlab
```

Dependencies are currently unpinned, and a clean-environment reproduction has not yet been verified.

### 2. Configure the dataset

This project uses **[Liberals vs Conservatives on Reddit [13000 posts]](https://www.kaggle.com/datasets/neelgajare/liberals-vs-conservatives-on-reddit-13000-posts)**, published on Kaggle by **neelgajare**. Credit for the source dataset belongs to its original contributor.

Download the dataset from Kaggle and place the classification CSV at `data/reddit_dr.csv` (rename it if necessary). The local copy used for the recorded results contains 12,854 posts. The main workflow requires these columns:

| Column | Use |
| --- | --- |
| `Title` | Post title, combined with the body |
| `Text` | Post body |
| `Political Lean` | Target label: `Liberal` or `Conservative` |

The local dataset also includes post metadata, but the classifiers use the text and derived features as predictors.

The notebooks currently contain an absolute local data path. Before running the main notebook, replace its `pd.read_csv(...)` line with:

```python
df = pd.read_csv("data/reddit_dr.csv")
```

Run Jupyter from the repository root so this relative path resolves correctly. Consult the Kaggle listing for dataset documentation and applicable terms; the label-assignment procedure and redistribution license have not yet been verified for this README.

### 3. Run the main notebook

Open `full_classifier_virtue_violation_patched.ipynb`, select the **Python (PBL2)** kernel, and run cells in order. Start with the feature extraction and three-model comparison; the later classifier sweep and ablation sections add more extensive experiments.

The first run downloads pretrained models and requires internet access. RoBERTa embedding extraction selects CUDA, Apple MPS, or CPU when available; the sentiment pipeline is configured for CPU. Feature extraction and the larger grid searches can take substantial time. Running the notebook also writes the generated feature table and, in the classifier sweep, the results CSV.

## Limitations and next steps

- **Label interpretation:** Predictions reflect the dataset's two labels. They do not establish an author's personal political beliefs or capture the full range of political views.
- **Generalization:** A random post-level split does not establish performance on unseen communities or future posts. Grouped and temporal evaluation would provide stronger evidence.
- **Moral-language features:** Prototype similarities are heuristic text features, not validated measurements of a person's values. Their usefulness depends on the descriptions and embedding model.
- **Text truncation:** RoBERTa embeddings use at most 256 tokens per post; the sentiment pipeline uses at most 512, so long posts can lose information.
- **Model comparison:** Many experiments reuse the same test split. A new untouched evaluation set would help validate conclusions after model selection.
- **Reproducibility:** Next steps include verifying the source dataset's labeling procedure, pinning dependencies, replacing local paths, and verifying the main notebook from a clean kernel.
