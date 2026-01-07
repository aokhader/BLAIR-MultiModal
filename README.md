# 🌟 **Multi-Modal Next-Item Recommendation using BLAIR-MM (Text + Image Embeddings)**


**Team Members:** _Will L., Derek P., Abdulaziz K., Mustafa H._  
**Instructor:** _Julian McAuley_

---

# 🚀 **Project Overview**

This project explores a **next-item recommendation task** using a **multi-modal embedding model** inspired by **BLAIR (Bridging Language and Items for Retrieval)**.

We extend BLAIR by incorporating **image embeddings** through the **CLIP image encoder**, creating a multimodal item representation we call **BLAIR-MM**.

Our predictive task:

> **Given a user’s interaction history, predict the next item they will interact with.**

We evaluate BLAIR-MM by integrating it into classic recommender models from DSC 256—primarily **Matrix Factorization (MF)**—and compare it to strong baseline models.

---

# 📚 **1. Predictive Task Definition**

### 🎯 Task

Predict the next item a user will consume, given their chronological interaction sequence:

\[
S*u = [i_1, i_2, ..., i_t] \quad \rightarrow \quad i*{t+1}
\]

### 📈 Evaluation Metrics

-   _**Recall@10 / Recall@50**_
-   _**AUC**_
-   _**Nearest-neighbor inspection of embeddings**_
-   _**Cold-start evaluation**_

### 🧪 Baselines

1. _**Popularity baseline**_
2. _**Last-item transition (Markov-like)**_
3. _**Matrix Factorization (MF)**_
4. _**item2vec (skip-gram)**_
5. _**Our model: BLAIR-MM (CLIP + BLAIR)** integrated into MF_

### ✔ Validity Checks

-   _Cold-start behavior_
-   _Sequence sanity checks_
-   _Qualitative nearest neighbors_
-   _Baseline comparisons_

---

# 🔍 **2. Dataset, EDA, and Preprocessing**

### 📦 Dataset

We use an Amazon-style dataset including:

-   _Product **text metadata**_
-   _Product **images**_
-   _**User-item interactions** with timestamps_

### 🧹 Preprocessing

-   _Text tokenization (BLAIR-compatible)_
-   _Image resizing → CLIP format_
-   _User sequence construction_
-   _Train/val/test split using leave-one-out_

### 📊 EDA Components

-   _Popularity distribution_
-   _Item frequency long-tail visualization_
-   _Sequence length plots_
-   _Text length histograms_
-   _Sample text + image previews_

---

# 🧠 **3. Modeling**

## 🎯 Problem Formulation

Next-item prediction as a _ranking_ task.

---

## 🏗 **Model Architecture — BLAIR-MM**

### **Text Encoder — BLAIR**

-   _RoBERTa-based encoder_
-   _Extracts 768-d CLS embedding_

### **Image Encoder — CLIP**

-   _ViT-B/32 backbone_
-   _Produces 512–768-d image embedding_

### **Fusion Module**

-   _Concatenate: ([text | image])_
-   _Feed through MLP → **768-d fused item embedding**_

### **Contrastive Objective (InfoNCE)**

Align:

-   _**context text embedding** (from user history)_
-   _**fused item embedding**_

---

## 🔮 **Downstream Model (Recommender)**

We plug the BLAIR-MM item embeddings into:

-   **Matrix Factorization (MF)** for personalized scoring  
    \\[
    \text{score}(u,i) = p_u^\top e_i^{\text{BLAIR-MM}}
    \\]

This keeps our sequential modeling simple and aligned with DSC 256.
For details on how to run the BLAIR + CLIP model, the scripts' details are found [here](./blair/README.md).

---

# 📊 **4. Evaluation**

### Metrics

-   _Recall@10 / Recall@50_
-   _AUC_
-   _Cold-start analysis_
-   _Embedding nearest neighbor visualization_

### Results Table

![model_results](./BLAIR-CLIP-dataset/model_results.png)

---

# 📚 **5. Related Work**

### Classical Recommender Models

-   _Matrix Factorization_
-   _Bayesian Personalized Ranking (BPR)_
-   _First-order sequence models (last-item transitions)_

### Text-based Retrieval Methods

-   _TF-IDF retrieval_
-   _item2vec (Skip-Gram)_
-   _Transformer text encoders_

### Multi-Modal Recommendation

-   _VBPR_
-   _DeepStyle_
-   _CLIP-based retrieval_
-   **_BLAIR (text-only embedding model)_**

### Our Contribution

-   _First multimodal extension of BLAIR using CLIP_
-   _Fusion of text + image for item representations_
-   _Sequential evaluation via next-item prediction_

---

# 📁 **Project Structure Highlights**

```
project/
│
├── README.md
├── model_showcase.html                # notebook detailing the modeling process
├── baseline_utils.py                  # utility file for splitting data and evaluating baseline models
│
├── baselines/
│   ├── baseline_mf.py                 # MF model class definition
│   ├── baseline_tfidf.py              # TF-IDF model class definition
│   ├── run_baselines.py               # trains and evaluates the baseline models with and without images
│  
├── encoders/
│   ├── clip_encoder.py                # encoder used for CLIP model
│
├── blair/
│   ├── multimodal/ 
│   ├── blair_clip.py                  # BLAIR-MM class definition
│   ├── sample_multimodal_data.py      # preprocess dataset for MM model
```

---

# 🎉 **Conclusion**

BLAIR-MM produces **multimodal item embeddings** by combining text (BLAIR) and image (CLIP) signals.  
When integrated into MF, these embeddings significantly outperform classical baselines in next-item recommendation, especially under cold-start conditions.
