# CAS2105 Homework 6: AG News Headline Classification
**Author:** Jein Seo (2022148090)

---

## 1. Introduction

This project implements a complete but lightweight **AI pipeline** for the **AG News headline classification** task. The objective is to classify short news headlines into one of four categories:

- **World**
- **Sports**
- **Business**
- **Sci/Tech**

Rather than training large, resource-intensive models, this assignment focuses on comparing:

1. A **naïve keyword-matching baseline**, and  
2. A **semantic embedding model** (MiniLM) combined with a logistic regression classifier.

News headlines are short and often ambiguous. Relying solely on keyword heuristics frequently leads to misclassification, making this task an excellent showcase of the power of semantic embeddings.

---

## 2. Task Definition

### Task
Predict the topic category of a single English news headline.

### Formal Definition
- **Input:** A raw headline string  
- **Output:** A label *y* ∈ {0, 1, 2, 3}  
- **Goal:** Learn a function *f(x) → y* that generalizes well to unseen headlines.

### Success Criteria
We consider the project successful if:

- The AI pipeline achieves **substantially higher accuracy** than the baseline.
- It improves the **macro F1-score**, which treats all four classes equally.

### Motivation
News headline classification is widely used in:

- Recommendation algorithms  
- Topic-aware ranking  
- News aggregation  
- Content moderation and filtering  

Headlines require semantic interpretation, making this a suitable domain to compare naïve rules against embedding-based learning.

---

## 3. Methods

We evaluate two approaches:

1. **Naïve keyword-based baseline**
2. **MiniLM-based semantic embedding pipeline**

---

### 3.1 Naïve Baseline

The baseline uses a keyword dictionary for each class. It scans the headline for predefined keywords and predicts the corresponding class. If no keyword is detected, it defaults to **World**.

#### Why This Baseline?
It serves as a simple, intuitive heuristic system. Many news categories do correlate with distinctive words (e.g., “tournament” → Sports).

#### Limitations
- Fails when headlines contain **ambiguous** terms  
- Confuses overlapping terminology (e.g., “Tech company reports earnings” → keyword "Tech" incorrectly predicts Sci/Tech)  
- Cannot interpret meaning  
- Struggles with short headlines lacking explicit signals

#### Baseline Implementation (excerpt)
    ```python
    def baseline_predict(text):
        text = text.lower()
        for label, keywords in keyword_map.items():
            if any(k in text for k in keywords):
                return label
        return 0  # fallback: "World"

### 3.2 AI Pipeline

The improved AI pipeline uses **pre-trained semantic embeddings** and a lightweight **logistic regression classifier** to perform news topic classification. Unlike the baseline, which relies only on literal keyword matching, the embedding pipeline captures the deeper semantic meaning of each headline.

This allows it to correctly classify headlines that do not contain any obvious topic-specific keywords or contain ambiguous words used differently across topics.

---

#### Models Used

- **Embedding model:** `all-MiniLM-L6-v2` from SentenceTransformers  
  - Produces 384-dimensional embeddings  
  - Lightweight and fast (CPU-friendly)  
  - Trained using deep self-attention distillation (MiniLM)

- **Classifier:** Logistic Regression (`scikit-learn`)
  - Works well with dense embeddings  
  - Fast to train  
  - Provides strong linear separation of semantic clusters  

---

#### Pipeline Stages

1. **Text Preprocessing**
   - Lowercasing
   - Removing extra whitespace  
   *No stemming or tokenization needed—the embedding model handles raw text well.*

2. **Embedding**
   Each headline string is passed through MiniLM to generate a **semantic vector**.
   ```python
   model = SentenceTransformer("all-MiniLM-L6-v2")
   X_train_embed = model.encode(X_train)

3. **Classifier Training**
   The embeddings are fed into logistic regression:

4. **Prediction**
   Test headlines are embedded and classified:

#### Why This Pipeline?

The semantic embedding pipeline was chosen because it addresses nearly all limitations of the naïve keyword baseline. Headlines are often short, metaphorical, and lacking explicit keywords. Therefore, a system that understands semantic meaning—rather than surface-level word matching—is necessary.

MiniLM embeddings capture deeper contextual relationships such as:

- Financial terminology → **Business**  
- Scientific discovery wording → **Sci/Tech**  
- International conflict cues → **World**  
- Competition, scores, team references → **Sports**

Because the embeddings are high-dimensional representations of meaning, the classifier can easily separate classes in this semantic space. This makes the pipeline robust to variations in phrasing, vocabulary choice, and ambiguity.

Additionally, MiniLM is extremely lightweight, allowing the entire pipeline to run smoothly on CPU with no fine-tuning, making it ideal for a small student project.

---

## 4. Experiments

### 4.1 Dataset

The experiments use the **AG News** dataset from Hugging Face. This dataset contains news headlines categorized into four balanced classes:

1. **World**
2. **Sports**
3. **Business**
4. **Sci/Tech**

To keep the computation efficient:

- **Train set:** 3,000 examples (750 per class)  
- **Test set:** 1,000 examples (250 per class)

This subset size is large enough to demonstrate meaningful improvements between models while still being fast to embed and train on a CPU.

#### Preprocessing
Minimal text preprocessing was applied:

- lowercasing  
- whitespace normalization  

No stemming, lemmatization, or stopword removal was used. This ensures the embedding model receives intact linguistic context.

---

### 4.2 Metrics

Two evaluation metrics were used:

- **Accuracy:** overall proportion of correct predictions  
- **Macro F1-score:** average of F1-scores across all four classes, treating each class equally  

Macro F1 is essential when analyzing multi-class settings, especially when some topics have subtle distinguishing features.

---

### 4.3 Results

| Method | Accuracy | Macro F1 |
|--------|----------|----------|
| **Naïve Baseline** | 0.537 | 0.524 |
| **Embedding Pipeline (MiniLM)** | 0.873 | 0.874 |

### Interpretation

The embedding pipeline dramatically outperforms the baseline, improving:

- **Accuracy by 33.6 percentage points**
- **Macro F1 by 35 percentage points**

This demonstrates that:

- Keyword-based heuristics are not reliable for semantic tasks.  
- MiniLM embeddings form well-separated clusters for each topic.  
- Even a simple linear classifier (logistic regression) is sufficient when the representation is strong.

The baseline frequently misclassifies Business vs. Sci/Tech headlines due to shared terminology. The embedding model resolves this ambiguity by understanding **context** instead of keyword co-occurrence.

---

### 4.4 Qualitative Examples

Below are examples where the embedding model succeeds while the baseline fails.

#### Example 1  
**Headline:**  
*“General Mills goes whole grains…”*

- **True Label:** Business  
- **Baseline Prediction:** World  
- **Embedding Prediction:** Business (Correct!)

**Explanation:**  
The headline lacks explicit business-related keywords.  
The baseline misclassifies it due to missing signals, whereas the embedding model infers the **corporate/product** context.

---

#### Example 2  
**Headline:**  
*“Profit plunges at International Game Tech…”*

- **True Label:** Business  
- **Baseline Prediction:** Sci/Tech  
- **Embedding Prediction:** Business (Correct!)

**Explanation:**  
The baseline is misled by the keyword **“Tech”**.  
The embedding model recognizes the headline as a **financial performance report**, not science/technology news.

---

## 5. Reflection and Limitations

This project illustrates the stark difference between simplistic heuristics and modern embedding-based NLP methods.

### What Worked Well

- **MiniLM embeddings performed exceptionally**, even without fine-tuning  
- Logistic regression trained in seconds and generalized well  
- Balanced dataset made evaluation straightforward  
- Semantic representations captured nuances of news topics  

---

### Challenges Encountered

- Building a fair keyword list for the baseline was surprisingly difficult  
- Business and Sci/Tech articles often overlap in terminology  
- Ensuring proper labeling for ambiguous headlines required careful interpretation  
- Some short headlines remained difficult even for the embedding model  

---

### Future Improvements

With more time or computational resources, several enhancements could be explored:

- Fine-tuning DistilBERT or MiniLM on the AG News dataset  
- Increasing the training dataset from 3k to 10k+ samples  
- Analyzing confusion matrices to isolate common failure modes  
- Adding entity recognition (NER) as additional features  
- Using non-linear classifiers like SVM or a small MLP  

---

## References

[1] Zhang, X., Zhao, J., & LeCun, Y. (2015). **Character-level Convolutional Networks for Text Classification.** *Advances in Neural Information Processing Systems (NeurIPS).*  
Paper: https://papers.nips.cc/paper_files/paper/2015/hash/250cf8b51c773f3f8dc8b4be867a9a02-Abstract.html  
Dataset (AG News): https://huggingface.co/datasets/ag_news

[2] Lhoest, Q., et al. (2021). **Datasets: A Community Library for Natural Language Processing.** *Hugging Face.*  
Documentation: https://huggingface.co/docs/datasets  
Library: https://github.com/huggingface/datasets

[3] Reimers, N., & Gurevych, I. (2019). **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.** *EMNLP.*  
Paper: https://arxiv.org/abs/1908.10084  
Model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

[4] Pedregosa, F., et al. (2011). **Scikit-learn: Machine Learning in Python.** *Journal of Machine Learning Research.*  
Documentation: https://scikit-learn.org

[5] Wang, W., et al. (2020). **MiniLM: Deep Self-Attention Distillation for Task-Agnostic NLP.** *ACL.*  
Paper: https://arxiv.org/abs/2002.10957
