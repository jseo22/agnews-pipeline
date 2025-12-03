# Mini AI Pipeline Project — AG News Headline Classification

## 1. Introduction

This project builds a small AI pipeline for **news headline classification** using the AG News dataset. The task is a **4-class classification problem**:

- **World**
- **Sports**
- **Business**
- **Sci/Tech**

The pipeline includes:

1. A **naïve keyword-based baseline**
2. A **semantic embedding model using SentenceTransformers**
3. Evaluation and comparison of both approaches

The goal is to understand the AI development workflow and demonstrate how pre-trained embeddings outperform simple heuristics.

---

## 2. Dataset

- **Dataset:** AG News (Hugging Face)
- **Split:**
  - Train: 3,000 headlines (750 per class)
  - Test: 1,000 headlines (balanced across 4 classes)
- **Features:**
  - `text` — news headline
  - `label` — category index (0–3)

### Preprocessing

- Lowercasing
- Removing line breaks
- Normalizing whitespace

No aggressive normalization was used.

---

## 3. Methods

### 3.1 Baseline — Keyword Matching

A simple keyword lookup for domain-specific words:
- “government”, “president” → World  
- “team”, “win”, “league” → Sports  
- “market”, “stocks”, “profit” → Business  
- “software”, “technology”, “devices” → Sci/Tech  

**Strengths**
- Simple, cheap, interpretable  
- Works for obvious cases  

**Weaknesses**
- Cannot understand meaning  
- Fails when keywords overlap across domains  
- Misclassifies headlines with ambiguous vocabulary  

---

### 3.2 Embedding Pipeline

Model:
- **Encoder:** `all-MiniLM-L6-v2` (SentenceTransformers)
- **Representation:** 384-dimensional sentence embedding
- **Classifier:** Logistic Regression

Pipeline steps:

1. Preprocess text  
2. Convert headline → dense embedding  
3. Train classifier on embeddings  
4. Predict labels on new text  

This pipeline captures semantic relationships between words, enabling far better generalization.

---

## 4. Results

### 4.1 Quantitative Results

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Keyword Baseline | **0.537** | **0.524** |
| Embedding Pipeline | **0.873** | **0.874** |

### Interpretation
The embedding system improves performance by:

- **+33.6% accuracy**
- **+35% macro F1**

Across all classes, embedding-based classification is significantly more robust:

- **Sports** improved from F1 = 0.65 → 0.96  
- **Sci/Tech** improved from F1 = 0.42 → 0.84  
- **Business** improved from F1 = 0.46 → 0.81  

---

## 4.2 Qualitative Examples

These real test cases highlight the difference between the baseline and the embedding model.

---

### **Example 1**
**Text:**  
“General Mills goes whole grains… General Mills announced plans… to start using healthier whole grains…”

**True label:** Business  
**Baseline:** World  
**Embedding:** Business (correct)

**Reasoning:**  
The baseline triggers on words like “General,” “plans,” and “announced,” which appear frequently in World news. The embedding model correctly identifies this as a corporate product announcement.

---

### **Example 2**
**Text:**  
“Profit Plunges at International Game Tech… said profit fell 50 percent due to a tax adjustment.”

**True label:** Business  
**Baseline:** Sci/Tech  
**Embedding:** Business (correct)

**Reasoning:**  
The baseline fixates on “Tech,” incorrectly placing it into Sci/Tech.  
Embeddings understand the headline describes **financial performance**, not technology.

---

## 5. Reflection

This project demonstrates:

- Keyword baselines can capture obvious patterns but fail on semantic nuance.
- Embeddings dramatically improve classification through contextual understanding.
- Real-world text classification requires models that go beyond surface forms.
- Even lightweight models (MiniLM) can achieve high accuracy with simple pipelines.

### Future improvements
- Fine-tuning a transformer model (e.g., DistilBERT)
- Larger training samples (10k+ headlines)
- Confusion-matrix-driven error analysis
- Class-specific threshold tuning

---

## 6. Conclusion

This mini pipeline project shows a clear, measurable improvement from naive heuristics to modern AI methods. Semantic embeddings provide a powerful yet accessible way to build robust text classifiers, even with limited data and simple modeling techniques.

