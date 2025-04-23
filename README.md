# 🧠  Language Representations 

The **Language Representations** track. This repository contains end-to-end experiments, code, and insights on learning and evaluating high-quality multilingual embeddings.

---

## 🧩 Task Overview

The task is split into two parts:

### **🔹 Part 1: Language Representation Analysis**
- **Goal:** Analyze and visualize how word embeddings (like Word2Vec, FastText) capture semantic relationships.
- **Techniques:** 
  - Nearest neighbor similarity
  - PCA / t-SNE visualizations
  - Embedding analogies (e.g., king - man + woman = queen)

---

### **🔸 Part 2: Cross-lingual Alignment**
- **Goal:** Align monolingual word embeddings of English and Hindi into a shared vector space.
- **Steps:**
  - Preprocess and tokenize Hindi corpus using `indic-nlp-library`
  - Train Hindi embeddings from scratch or use pretrained FastText
  - Load English embeddings from FastText or use custom-trained vectors
  - Apply **Procrustes analysis** (orthogonal transformation) for alignment
  - Evaluate alignment quality with:
    - Precision@k
    - Mean Reciprocal Rank (MRR)
    - Word similarity correlation

---

## 💻 Repository Structure

```
.
├── Data/                            # Contains Hindi corpus files
├── Embeddings/                     # Custom trained embeddings
├── Models/                         # Pickled word embedding models
├── notebooks/                      # Jupyter notebooks for experimentation
├── utils/                          # Helper scripts (e.g., tokenization, preprocessing)
├── bilingual_dictionary.txt        # Ground truth dictionary for alignment evaluation
├── align_embeddings.py             # Procrustes-based alignment script
├── evaluate_alignment.py           # Evaluation (Precision@k, MRR)
├── visualize_embeddings.py         # t-SNE, PCA visualization scripts
└── README.md
```

---

## 🚀 Getting Started

### ✅ Prerequisites

```bash
pip install numpy pandas matplotlib scikit-learn gensim tqdm
pip install git+https://github.com/anoopkunchukuttan/indic_nlp_library.git
```

### ✅ Setup Indic NLP Resources

Download [Indic NLP resources](https://anoopkunchukuttan.github.io/indic_nlp_library/downloads.html) and set the path in your code:
```python
INDIC_RESOURCES_PATH = "/path/to/indic_nlp_resources"
```

---

## 🔧 Running the Project

### 👉 most of the experimentation parts are in jupyter notebooks

### 👉 Step 1: Preprocess Hindi Corpus
```bash
python utils/preprocess_hindi.py
```

### 👉 Step 2: Train Hindi Embeddings (Optional)
```bash
python utils/train_cooccurrence_embeddings.py
```


## 📎 Download Project (ZIP)

Click below to download the project directly as a zip:

🔗 [Download this repository as ZIP](https://github.com/raviteja-bommireddy/PreCog_/archive/refs/heads/main.zip)

---

## ✍️ Author

**Ravi Teja Bommireddy**  
B.Tech @ IIITDM Kancheepuram  
AI + NLP Enthusiast | Research Aspirant

---

## 🧠 Acknowledgements

- [FastText](https://fasttext.cc/)
- [Indic NLP Library](https://anoopkunchukuttan.github.io/indic_nlp_library/)
- [Precog Research Group](https://precog.iiit.ac.in/)
- 

---

## 📬 Contact

For queries or feedback, feel free to open an [Issue](https://github.com/raviteja-bommireddy/PreCog_/issues) 
or reach me via [LinkedIn](https://www.linkedin.com/in/raviteja-bommireddy/) 
or via mail : cs23b2011@iiitdm.ac.in