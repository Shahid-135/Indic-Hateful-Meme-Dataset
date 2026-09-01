# 📊 IndicHM Dataset

<p align="center">
  <strong>A Large-Scale Multilingual Multimodal Benchmark for Hateful Meme Detection in Indic Languages</strong>
</p>

<p align="center">
  <a href="#-dataset-access">🔒 Dataset Access</a> •
  <a href="#-overview">📖 Overview</a> •
  <a href="#-motivation">🎯 Motivation</a> •
  <a href="#-dataset-statistics">📊 Statistics</a> •
  <a href="#-tasks">⚡ Tasks</a> •
  <a href="#-models-and-methodology">🤖 Models</a> •
  <a href="#-reproducibility">🔄 Reproducibility</a> •
  <a href="#-citation">📝 Citation</a>
</p>

---

## 🔒 Dataset Access

> **⚠️ Important:** Access to the dataset is restricted to **academic and research purposes only.**

### Requirements

- ✅ **Non-Commercial Use:** Strictly for academic research. No commercial use permitted.
- ✅ **No Redistribution:** Dataset must not be shared, republished, or redistributed.
- ✅ **Platform Compliance:** Must adhere to Pinterest and Reddit Developer Agreements.
- ✅ **Academic Integrity:** Usage must comply with ethical standards and privacy guidelines.

### How to Request Access

1. **Use your institutional email address** (personal emails will be rejected)
2. Fill out the [Consent Form](https://docs.google.com/document/d/1G1kSiTy53pZ5KZOVCoTQCKTJ415cYCTL/edit?usp=sharing&ouid=107493269096531134179&rtpof=true&sd=true)
3. Submit the completed form via the [Dataset Request Form](https://forms.gle/CLT4pM4NTKZYWCEG7)
4. Upon approval, you'll receive download instructions via email

---

## 📖 Overview

**IndicHM** is a comprehensive multilingual multimodal benchmark for hateful meme detection, featuring:

- **15,004** hateful and non-hateful memes
- **5 hate categories** for fine-grained classification
- **3 languages:** Hindi, English, and Bengali
- **Code-mixed content** in Hindi and Bengali subsets

### Key Features

| Feature | Description |
|---------|-------------|
| 🌐 **Multilingual** | Hindi, English, Bengali |
| ⚖️ **Balanced** | Nearly equal distribution across languages and classes |
| 🔀 **Code-Mixed** | Hindi-English and Bengali-English code-mixed memes |
| 📊 **Multi-Task** | Binary hate detection + Multiclass hate categorization |

---

## 🎯 Motivation

Hateful memes derive their meaning from the **interaction between textual and visual cues**, making them challenging for unimodal approaches. Despite their widespread use on social media, **multilingual multimodal meme data—especially for Indic languages—remains scarce**, with most existing resources being English-centric.

This limitation is particularly critical in India, a linguistically diverse country with multiple regional languages, where online content is often multilingual and code-mixed. **IndicHM** addresses this gap by providing a dedicated benchmark for robust hate detection research in Indic languages.

---

## 📊 Dataset Statistics

### Task 1: Binary Hate Detection

**Split Size**

| Split | Samples | Percentage |
|-------|--------:|-----------:|
| Train | 12,003 | 80.00% |
| Val   | 1,500  | 10.00% |
| Test  | 1,501  | 10.00% |
| **Total** | **15,004** | **100%** |

**Label Distribution**

| Set | Non-Hate | Hate |
|-----|---------:|-----:|
| Train | 6,000 (49.99%) | 6,003 (50.01%) |
| Val   | 750 (50.00%)   | 750 (50.00%)   |
| Test  | 750 (49.97%)   | 751 (50.03%)   |
| **Overall** | **7,500 (49.99%)** | **7,504 (50.01%)** |

**Language-wise Distribution Across Splits**

| Split | Language | Count | Percentage (%) |
|------|--------|-----:|------:|
| Train | Hindi   | 4,007 | 33.38 |
| Train | English | 3,996 | 33.29 |
| Train | Bengali | 4,000 | 33.33 |
| Validation | Hindi   | 501 | 33.40 |
| Validation | English | 499 | 33.27 |
| Validation | Bengali | 500 | 33.33 |
| Test | Hindi   | 501 | 33.38 |
| Test | English | 500 | 33.31 |
| Test | Bengali | 500 | 33.31 |

### Task 2: Multiclass Hate Categorization

**Split Size**

| Split | Samples | Percentage |
|-------|--------:|-----------:|
| Train | 6,001 | 79.97% |
| Val   | 750   | 9.99%  |
| Test  | 753   | 10.03% |
| **Total** | **7,504** | **100%** |

**Category Distribution**

| Category | Train | Val | Test | Overall |
|----------|------:|----:|-----:|--------:|
| Gender Based | 1,592 | 199 | 199 | 1,990 |
| Others       | 949  | 119 | 119 | 1,187 |
| Political    | 1,175| 147 | 147 | 1,469 |
| Racism       | 843  | 105 | 107 | 1,055 |
| Religion     | 1,442| 180 | 181 | 1,803 |

**Language-wise Distribution Across Splits**

| Split | Language | Count | Percentage (%) |
|------|--------|-----:|------:|
| Train | Hindi   | 2,006 | 33.43 |
| Train | English | 1,995 | 33.24 |
| Train | Bengali | 2,000 | 33.33 |
| Validation | Hindi   | 251 | 33.47 |
| Validation | English | 249 | 33.20 |
| Validation | Bengali | 250 | 33.33 |
| Test | Hindi   | 252 | 33.47 |
| Test | English | 251 | 33.33 |
| Test | Bengali | 250 | 33.20 |

### Overall Language-wise Distribution

| Language | Total Samples | Hate | Non-Hate | Gender | Religion | Political | Racism | Others |
|----------|--------------:|-----:|---------:|-------:|---------:|----------:|-------:|-------:|
| Hindi    | 5,009 | 2,509 | 2,500 | 680 | 608 | 560 | 223 | 438 |
| English  | 4,995 | 2,495 | 2,500 | 810 | 695 | 409 | 332 | 249 |
| Bengali  | 5,000 | 2,500 | 2,500 | 500 | 500 | 500 | 500 | 500 |
| **Total** | **15,004** | **7,504** | **7,500** | **1,990** | **1,803** | **1,469** | **1,055** | **1,187** |

### Mixed-Code Statistics

| Language | Total Samples | Code-Mixed | Code-Mixed % | Script-Switched | Transliterated |
|----------|--------------:|-----------:|-------------:|----------------:|---------------:|
| Hindi    | 5,009 | 2,055 | 41.03% | 582 | 1,473 |
| Bengali  | 5,000 | 1,105 | 21.10% | 1,046 | 59 |
| English  | 4,995 | — | — | — | — |
| **Total** | **15,004** | **3,160** | **~21.06%** | **1,628** | **1,532** |

---

## ⚡ Tasks

### Task 1: Binary Hate Detection
Determines whether a given meme is **Hateful** or **Non-Hateful**.

### Task 2: Multiclass Hate Categorization
For samples labeled as hate, predicts one of five categories:

- 👥 **Gender-based Hate**
- ⛪ **Religious Hate**
- 🏛️ **Political Hate**
- 🚫 **Racist Hate**
- 📌 **Other Forms of Hate**

---

## 🤖 Models and Methodology

### Proposed Framework: **IndicHMNet**

We propose a **task-aware multilingual multimodal framework** for hateful meme detection that:

1. Jointly models visual and textual features while explicitly accounting for cross-modal incongruity
2. Uses a **pretrained SigLIP2 encoder** for aligned image–text representations
3. Employs a **disagreement-aware gated mechanism** to adaptively emphasize the dominant modality
4. Optimizes hate detection using **supervised classification** augmented with **label-conditioned retrieval-guided contrastive learning**
5. Trains target classification independently using cross-entropy loss

### Baseline Models

| Modality | Models |
|----------|--------|
| **Text** | MuRIL, mBERT, XLM-R |
| **Vision** | ViT, ConvNeXt |
| **Multimodal** | SigLIP2-based framework |

---

## 📁 Repository Structure
# 📊 IndicHM Dataset

<p align="center">
  <strong>A Large-Scale Multilingual Multimodal Benchmark for Hateful Meme Detection in Indic Languages</strong>
</p>

<p align="center">
  <a href="#-dataset-access">🔒 Dataset Access</a> •
  <a href="#-overview">📖 Overview</a> •
  <a href="#-motivation">🎯 Motivation</a> •
  <a href="#-dataset-statistics">📊 Statistics</a> •
  <a href="#-tasks">⚡ Tasks</a> •
  <a href="#-models-and-methodology">🤖 Models</a> •
  <a href="#-reproducibility">🔄 Reproducibility</a> •
  <a href="#-citation">📝 Citation</a>
</p>

---

## 🔒 Dataset Access

> **⚠️ Important:** Access to the dataset is restricted to **academic and research purposes only.**

### Requirements

- ✅ **Non-Commercial Use:** Strictly for academic research. No commercial use permitted.
- ✅ **No Redistribution:** Dataset must not be shared, republished, or redistributed.
- ✅ **Platform Compliance:** Must adhere to Pinterest and Reddit Developer Agreements.
- ✅ **Academic Integrity:** Usage must comply with ethical standards and privacy guidelines.

### How to Request Access

1. **Use your institutional email address** (personal emails will be rejected)
2. Fill out the [Consent Form](https://docs.google.com/document/d/1G1kSiTy53pZ5KZOVCoTQCKTJ415cYCTL/edit?usp=sharing&ouid=107493269096531134179&rtpof=true&sd=true)
3. Submit the completed form via the [Dataset Request Form](https://forms.gle/CLT4pM4NTKZYWCEG7)
4. Upon approval, you'll receive download instructions via email

---

## 📖 Overview

**IndicHM** is a comprehensive multilingual multimodal benchmark for hateful meme detection, featuring:

- **15,004** hateful and non-hateful memes
- **5 hate categories** for fine-grained classification
- **3 languages:** Hindi, English, and Bengali
- **Code-mixed content** in Hindi and Bengali subsets

### Key Features

| Feature | Description |
|---------|-------------|
| 🌐 **Multilingual** | Hindi, English, Bengali |
| ⚖️ **Balanced** | Nearly equal distribution across languages and classes |
| 🔀 **Code-Mixed** | Hindi-English and Bengali-English code-mixed memes |
| 📊 **Multi-Task** | Binary hate detection + Multiclass hate categorization |

---

## 🎯 Motivation

Hateful memes derive their meaning from the **interaction between textual and visual cues**, making them challenging for unimodal approaches. Despite their widespread use on social media, **multilingual multimodal meme data—especially for Indic languages—remains scarce**, with most existing resources being English-centric.

This limitation is particularly critical in India, a linguistically diverse country with multiple regional languages, where online content is often multilingual and code-mixed. **IndicHM** addresses this gap by providing a dedicated benchmark for robust hate detection research in Indic languages.

---

## 📊 Dataset Statistics

### Task 1: Binary Hate Detection

**Split Size**

| Split | Samples | Percentage |
|-------|--------:|-----------:|
| Train | 12,003 | 80.00% |
| Val   | 1,500  | 10.00% |
| Test  | 1,501  | 10.00% |
| **Total** | **15,004** | **100%** |

**Label Distribution**

| Set | Non-Hate | Hate |
|-----|---------:|-----:|
| Train | 6,000 (49.99%) | 6,003 (50.01%) |
| Val   | 750 (50.00%)   | 750 (50.00%)   |
| Test  | 750 (49.97%)   | 751 (50.03%)   |
| **Overall** | **7,500 (49.99%)** | **7,504 (50.01%)** |

**Language-wise Distribution Across Splits**

| Split | Language | Count | Percentage (%) |
|------|--------|-----:|------:|
| Train | Hindi   | 4,007 | 33.38 |
| Train | English | 3,996 | 33.29 |
| Train | Bengali | 4,000 | 33.33 |
| Validation | Hindi   | 501 | 33.40 |
| Validation | English | 499 | 33.27 |
| Validation | Bengali | 500 | 33.33 |
| Test | Hindi   | 501 | 33.38 |
| Test | English | 500 | 33.31 |
| Test | Bengali | 500 | 33.31 |

### Task 2: Multiclass Hate Categorization

**Split Size**

| Split | Samples | Percentage |
|-------|--------:|-----------:|
| Train | 6,001 | 79.97% |
| Val   | 750   | 9.99%  |
| Test  | 753   | 10.03% |
| **Total** | **7,504** | **100%** |

**Category Distribution**

| Category | Train | Val | Test | Overall |
|----------|------:|----:|-----:|--------:|
| Gender Based | 1,592 | 199 | 199 | 1,990 |
| Others       | 949  | 119 | 119 | 1,187 |
| Political    | 1,175| 147 | 147 | 1,469 |
| Racism       | 843  | 105 | 107 | 1,055 |
| Religion     | 1,442| 180 | 181 | 1,803 |

**Language-wise Distribution Across Splits**

| Split | Language | Count | Percentage (%) |
|------|--------|-----:|------:|
| Train | Hindi   | 2,006 | 33.43 |
| Train | English | 1,995 | 33.24 |
| Train | Bengali | 2,000 | 33.33 |
| Validation | Hindi   | 251 | 33.47 |
| Validation | English | 249 | 33.20 |
| Validation | Bengali | 250 | 33.33 |
| Test | Hindi   | 252 | 33.47 |
| Test | English | 251 | 33.33 |
| Test | Bengali | 250 | 33.20 |

### Overall Language-wise Distribution

| Language | Total Samples | Hate | Non-Hate | Gender | Religion | Political | Racism | Others |
|----------|--------------:|-----:|---------:|-------:|---------:|----------:|-------:|-------:|
| Hindi    | 5,009 | 2,509 | 2,500 | 680 | 608 | 560 | 223 | 438 |
| English  | 4,995 | 2,495 | 2,500 | 810 | 695 | 409 | 332 | 249 |
| Bengali  | 5,000 | 2,500 | 2,500 | 500 | 500 | 500 | 500 | 500 |
| **Total** | **15,004** | **7,504** | **7,500** | **1,990** | **1,803** | **1,469** | **1,055** | **1,187** |

### Mixed-Code Statistics

| Language | Total Samples | Code-Mixed | Code-Mixed % | Script-Switched | Transliterated |
|----------|--------------:|-----------:|-------------:|----------------:|---------------:|
| Hindi    | 5,009 | 2,055 | 41.03% | 582 | 1,473 |
| Bengali  | 5,000 | 1,105 | 21.10% | 1,046 | 59 |
| English  | 4,995 | — | — | — | — |
| **Total** | **15,004** | **3,160** | **~21.06%** | **1,628** | **1,532** |

---

## ⚡ Tasks

### Task 1: Binary Hate Detection
Determines whether a given meme is **Hateful** or **Non-Hateful**.

### Task 2: Multiclass Hate Categorization
For samples labeled as hate, predicts one of five categories:

- 👥 **Gender-based Hate**
- ⛪ **Religious Hate**
- 🏛️ **Political Hate**
- 🚫 **Racist Hate**
- 📌 **Other Forms of Hate**

---

## 🤖 Models and Methodology

### Proposed Framework: **IndicHMNet**

We propose a **task-aware multilingual multimodal framework** for hateful meme detection that:

1. Jointly models visual and textual features while explicitly accounting for cross-modal incongruity
2. Uses a **pretrained SigLIP2 encoder** for aligned image–text representations
3. Employs a **disagreement-aware gated mechanism** to adaptively emphasize the dominant modality
4. Optimizes hate detection using **supervised classification** augmented with **label-conditioned retrieval-guided contrastive learning**
5. Trains target classification independently using cross-entropy loss

### Baseline Models

| Modality | Models |
|----------|--------|
| **Text** | MuRIL, mBERT, XLM-R |
| **Vision** | ViT, ConvNeXt |
| **Multimodal** | SigLIP2-based framework |

---

## 📁 Repository Structure
# 📊 IndicHM Dataset

<p align="center">
  <strong>A Large-Scale Multilingual Multimodal Benchmark for Hateful Meme Detection in Indic Languages</strong>
</p>

<p align="center">
  <a href="#-dataset-access">🔒 Dataset Access</a> •
  <a href="#-overview">📖 Overview</a> •
  <a href="#-motivation">🎯 Motivation</a> •
  <a href="#-dataset-statistics">📊 Statistics</a> •
  <a href="#-tasks">⚡ Tasks</a> •
  <a href="#-models-and-methodology">🤖 Models</a> •
  <a href="#-reproducibility">🔄 Reproducibility</a> •
  <a href="#-citation">📝 Citation</a>
</p>

---

## 🔒 Dataset Access

> **⚠️ Important:** Access to the dataset is restricted to **academic and research purposes only.**

### Requirements

- ✅ **Non-Commercial Use:** Strictly for academic research. No commercial use permitted.
- ✅ **No Redistribution:** Dataset must not be shared, republished, or redistributed.
- ✅ **Platform Compliance:** Must adhere to Pinterest and Reddit Developer Agreements.
- ✅ **Academic Integrity:** Usage must comply with ethical standards and privacy guidelines.

### How to Request Access

1. **Use your institutional email address** (personal emails will be rejected)
2. Fill out the [Consent Form](https://docs.google.com/document/d/1G1kSiTy53pZ5KZOVCoTQCKTJ415cYCTL/edit?usp=sharing&ouid=107493269096531134179&rtpof=true&sd=true)
3. Submit the completed form via the [Dataset Request Form](https://forms.gle/CLT4pM4NTKZYWCEG7)
4. Upon approval, you'll receive download instructions via email

---

## 📖 Overview

**IndicHM** is a comprehensive multilingual multimodal benchmark for hateful meme detection, featuring:

- **15,004** hateful and non-hateful memes
- **5 hate categories** for fine-grained classification
- **3 languages:** Hindi, English, and Bengali
- **Code-mixed content** in Hindi and Bengali subsets

### Key Features

| Feature | Description |
|---------|-------------|
| 🌐 **Multilingual** | Hindi, English, Bengali |
| ⚖️ **Balanced** | Nearly equal distribution across languages and classes |
| 🔀 **Code-Mixed** | Hindi-English and Bengali-English code-mixed memes |
| 📊 **Multi-Task** | Binary hate detection + Multiclass hate categorization |

---

## 🎯 Motivation

Hateful memes derive their meaning from the **interaction between textual and visual cues**, making them challenging for unimodal approaches. Despite their widespread use on social media, **multilingual multimodal meme data—especially for Indic languages—remains scarce**, with most existing resources being English-centric.

This limitation is particularly critical in India, a linguistically diverse country with multiple regional languages, where online content is often multilingual and code-mixed. **IndicHM** addresses this gap by providing a dedicated benchmark for robust hate detection research in Indic languages.

---

## 📊 Dataset Statistics

### Task 1: Binary Hate Detection

**Split Size**

| Split | Samples | Percentage |
|-------|--------:|-----------:|
| Train | 12,003 | 80.00% |
| Val   | 1,500  | 10.00% |
| Test  | 1,501  | 10.00% |
| **Total** | **15,004** | **100%** |

**Label Distribution**

| Set | Non-Hate | Hate |
|-----|---------:|-----:|
| Train | 6,000 (49.99%) | 6,003 (50.01%) |
| Val   | 750 (50.00%)   | 750 (50.00%)   |
| Test  | 750 (49.97%)   | 751 (50.03%)   |
| **Overall** | **7,500 (49.99%)** | **7,504 (50.01%)** |

**Language-wise Distribution Across Splits**

| Split | Language | Count | Percentage (%) |
|------|--------|-----:|------:|
| Train | Hindi   | 4,007 | 33.38 |
| Train | English | 3,996 | 33.29 |
| Train | Bengali | 4,000 | 33.33 |
| Validation | Hindi   | 501 | 33.40 |
| Validation | English | 499 | 33.27 |
| Validation | Bengali | 500 | 33.33 |
| Test | Hindi   | 501 | 33.38 |
| Test | English | 500 | 33.31 |
| Test | Bengali | 500 | 33.31 |

### Task 2: Multiclass Hate Categorization

**Split Size**

| Split | Samples | Percentage |
|-------|--------:|-----------:|
| Train | 6,001 | 79.97% |
| Val   | 750   | 9.99%  |
| Test  | 753   | 10.03% |
| **Total** | **7,504** | **100%** |

**Category Distribution**

| Category | Train | Val | Test | Overall |
|----------|------:|----:|-----:|--------:|
| Gender Based | 1,592 | 199 | 199 | 1,990 |
| Others       | 949  | 119 | 119 | 1,187 |
| Political    | 1,175| 147 | 147 | 1,469 |
| Racism       | 843  | 105 | 107 | 1,055 |
| Religion     | 1,442| 180 | 181 | 1,803 |

**Language-wise Distribution Across Splits**

| Split | Language | Count | Percentage (%) |
|------|--------|-----:|------:|
| Train | Hindi   | 2,006 | 33.43 |
| Train | English | 1,995 | 33.24 |
| Train | Bengali | 2,000 | 33.33 |
| Validation | Hindi   | 251 | 33.47 |
| Validation | English | 249 | 33.20 |
| Validation | Bengali | 250 | 33.33 |
| Test | Hindi   | 252 | 33.47 |
| Test | English | 251 | 33.33 |
| Test | Bengali | 250 | 33.20 |

### Overall Language-wise Distribution

| Language | Total Samples | Hate | Non-Hate | Gender | Religion | Political | Racism | Others |
|----------|--------------:|-----:|---------:|-------:|---------:|----------:|-------:|-------:|
| Hindi    | 5,009 | 2,509 | 2,500 | 680 | 608 | 560 | 223 | 438 |
| English  | 4,995 | 2,495 | 2,500 | 810 | 695 | 409 | 332 | 249 |
| Bengali  | 5,000 | 2,500 | 2,500 | 500 | 500 | 500 | 500 | 500 |
| **Total** | **15,004** | **7,504** | **7,500** | **1,990** | **1,803** | **1,469** | **1,055** | **1,187** |

### Mixed-Code Statistics

| Language | Total Samples | Code-Mixed | Code-Mixed % | Script-Switched | Transliterated |
|----------|--------------:|-----------:|-------------:|----------------:|---------------:|
| Hindi    | 5,009 | 2,055 | 41.03% | 582 | 1,473 |
| Bengali  | 5,000 | 1,105 | 21.10% | 1,046 | 59 |
| English  | 4,995 | — | — | — | — |
| **Total** | **15,004** | **3,160** | **~21.06%** | **1,628** | **1,532** |

---

## ⚡ Tasks

### Task 1: Binary Hate Detection
Determines whether a given meme is **Hateful** or **Non-Hateful**.

### Task 2: Multiclass Hate Categorization
For samples labeled as hate, predicts one of five categories:

- 👥 **Gender-based Hate**
- ⛪ **Religious Hate**
- 🏛️ **Political Hate**
- 🚫 **Racist Hate**
- 📌 **Other Forms of Hate**

---

## 🤖 Models and Methodology

### Proposed Framework: **IndicHMNet**

We propose a **task-aware multilingual multimodal framework** for hateful meme detection that:

1. Jointly models visual and textual features while explicitly accounting for cross-modal incongruity
2. Uses a **pretrained SigLIP2 encoder** for aligned image–text representations
3. Employs a **disagreement-aware gated mechanism** to adaptively emphasize the dominant modality
4. Optimizes hate detection using **supervised classification** augmented with **label-conditioned retrieval-guided contrastive learning**
5. Trains target classification independently using cross-entropy loss

### Baseline Models

| Modality | Models |
|----------|--------|
| **Text** | MuRIL, mBERT, XLM-R |
| **Vision** | ViT, ConvNeXt |
| **Multimodal** | SigLIP2-based framework |

---

## 📁 Repository Structure
# 📊 IndicHM Dataset

<p align="center">
  <strong>A Large-Scale Multilingual Multimodal Benchmark for Hateful Meme Detection in Indic Languages</strong>
</p>

<p align="center">
  <a href="#-dataset-access">🔒 Dataset Access</a> •
  <a href="#-overview">📖 Overview</a> •
  <a href="#-motivation">🎯 Motivation</a> •
  <a href="#-dataset-statistics">📊 Statistics</a> •
  <a href="#-tasks">⚡ Tasks</a> •
  <a href="#-models-and-methodology">🤖 Models</a> •
  <a href="#-reproducibility">🔄 Reproducibility</a> •
  <a href="#-citation">📝 Citation</a>
</p>

---

## 🔒 Dataset Access

> **⚠️ Important:** Access to the dataset is restricted to **academic and research purposes only.**

### Requirements

- ✅ **Non-Commercial Use:** Strictly for academic research. No commercial use permitted.
- ✅ **No Redistribution:** Dataset must not be shared, republished, or redistributed.
- ✅ **Platform Compliance:** Must adhere to Pinterest and Reddit Developer Agreements.
- ✅ **Academic Integrity:** Usage must comply with ethical standards and privacy guidelines.

### How to Request Access

1. **Use your institutional email address** (personal emails will be rejected)
2. Fill out the [Consent Form](https://docs.google.com/document/d/1G1kSiTy53pZ5KZOVCoTQCKTJ415cYCTL/edit?usp=sharing&ouid=107493269096531134179&rtpof=true&sd=true)
3. Submit the completed form via the [Dataset Request Form](https://forms.gle/CLT4pM4NTKZYWCEG7)
4. Upon approval, you'll receive download instructions via email

---

## 📖 Overview

**IndicHM** is a comprehensive multilingual multimodal benchmark for hateful meme detection, featuring:

- **15,004** hateful and non-hateful memes
- **5 hate categories** for fine-grained classification
- **3 languages:** Hindi, English, and Bengali
- **Code-mixed content** in Hindi and Bengali subsets

### Key Features

| Feature | Description |
|---------|-------------|
| 🌐 **Multilingual** | Hindi, English, Bengali |
| ⚖️ **Balanced** | Nearly equal distribution across languages and classes |
| 🔀 **Code-Mixed** | Hindi-English and Bengali-English code-mixed memes |
| 📊 **Multi-Task** | Binary hate detection + Multiclass hate categorization |

---

## 🎯 Motivation

Hateful memes derive their meaning from the **interaction between textual and visual cues**, making them challenging for unimodal approaches. Despite their widespread use on social media, **multilingual multimodal meme data—especially for Indic languages—remains scarce**, with most existing resources being English-centric.

This limitation is particularly critical in India, a linguistically diverse country with multiple regional languages, where online content is often multilingual and code-mixed. **IndicHM** addresses this gap by providing a dedicated benchmark for robust hate detection research in Indic languages.

---

## 📊 Dataset Statistics

### Task 1: Binary Hate Detection

**Split Size**

| Split | Samples | Percentage |
|-------|--------:|-----------:|
| Train | 12,003 | 80.00% |
| Val   | 1,500  | 10.00% |
| Test  | 1,501  | 10.00% |
| **Total** | **15,004** | **100%** |

**Label Distribution**

| Set | Non-Hate | Hate |
|-----|---------:|-----:|
| Train | 6,000 (49.99%) | 6,003 (50.01%) |
| Val   | 750 (50.00%)   | 750 (50.00%)   |
| Test  | 750 (49.97%)   | 751 (50.03%)   |
| **Overall** | **7,500 (49.99%)** | **7,504 (50.01%)** |

**Language-wise Distribution Across Splits**

| Split | Language | Count | Percentage (%) |
|------|--------|-----:|------:|
| Train | Hindi   | 4,007 | 33.38 |
| Train | English | 3,996 | 33.29 |
| Train | Bengali | 4,000 | 33.33 |
| Validation | Hindi   | 501 | 33.40 |
| Validation | English | 499 | 33.27 |
| Validation | Bengali | 500 | 33.33 |
| Test | Hindi   | 501 | 33.38 |
| Test | English | 500 | 33.31 |
| Test | Bengali | 500 | 33.31 |

### Task 2: Multiclass Hate Categorization

**Split Size**

| Split | Samples | Percentage |
|-------|--------:|-----------:|
| Train | 6,001 | 79.97% |
| Val   | 750   | 9.99%  |
| Test  | 753   | 10.03% |
| **Total** | **7,504** | **100%** |

**Category Distribution**

| Category | Train | Val | Test | Overall |
|----------|------:|----:|-----:|--------:|
| Gender Based | 1,592 | 199 | 199 | 1,990 |
| Others       | 949  | 119 | 119 | 1,187 |
| Political    | 1,175| 147 | 147 | 1,469 |
| Racism       | 843  | 105 | 107 | 1,055 |
| Religion     | 1,442| 180 | 181 | 1,803 |

**Language-wise Distribution Across Splits**

| Split | Language | Count | Percentage (%) |
|------|--------|-----:|------:|
| Train | Hindi   | 2,006 | 33.43 |
| Train | English | 1,995 | 33.24 |
| Train | Bengali | 2,000 | 33.33 |
| Validation | Hindi   | 251 | 33.47 |
| Validation | English | 249 | 33.20 |
| Validation | Bengali | 250 | 33.33 |
| Test | Hindi   | 252 | 33.47 |
| Test | English | 251 | 33.33 |
| Test | Bengali | 250 | 33.20 |

### Overall Language-wise Distribution

| Language | Total Samples | Hate | Non-Hate | Gender | Religion | Political | Racism | Others |
|----------|--------------:|-----:|---------:|-------:|---------:|----------:|-------:|-------:|
| Hindi    | 5,009 | 2,509 | 2,500 | 680 | 608 | 560 | 223 | 438 |
| English  | 4,995 | 2,495 | 2,500 | 810 | 695 | 409 | 332 | 249 |
| Bengali  | 5,000 | 2,500 | 2,500 | 500 | 500 | 500 | 500 | 500 |
| **Total** | **15,004** | **7,504** | **7,500** | **1,990** | **1,803** | **1,469** | **1,055** | **1,187** |

### Mixed-Code Statistics

| Language | Total Samples | Code-Mixed | Code-Mixed % | Script-Switched | Transliterated |
|----------|--------------:|-----------:|-------------:|----------------:|---------------:|
| Hindi    | 5,009 | 2,055 | 41.03% | 582 | 1,473 |
| Bengali  | 5,000 | 1,105 | 21.10% | 1,046 | 59 |
| English  | 4,995 | — | — | — | — |
| **Total** | **15,004** | **3,160** | **~21.06%** | **1,628** | **1,532** |

---

## ⚡ Tasks

### Task 1: Binary Hate Detection
Determines whether a given meme is **Hateful** or **Non-Hateful**.

### Task 2: Multiclass Hate Categorization
For samples labeled as hate, predicts one of five categories:

- 👥 **Gender-based Hate**
- ⛪ **Religious Hate**
- 🏛️ **Political Hate**
- 🚫 **Racist Hate**
- 📌 **Other Forms of Hate**

---

## 🤖 Models and Methodology

### Proposed Framework: **IndicHMNet**

We propose a **task-aware multilingual multimodal framework** for hateful meme detection that:

1. Jointly models visual and textual features while explicitly accounting for cross-modal incongruity
2. Uses a **pretrained SigLIP2 encoder** for aligned image–text representations
3. Employs a **disagreement-aware gated mechanism** to adaptively emphasize the dominant modality
4. Optimizes hate detection using **supervised classification** augmented with **label-conditioned retrieval-guided contrastive learning**
5. Trains target classification independently using cross-entropy loss

### Baseline Models

| Modality | Models |
|----------|--------|
| **Text** | MuRIL, mBERT, XLM-R |
| **Vision** | ViT, ConvNeXt |
| **Multimodal** | SigLIP2-based framework |

---

## 📁 Repository Structure

IndicHM/
├── Images/ # Dataset images
├── Code/ # Model, training, and evaluation code
│ ├── models/
│ ├── training/
│ └── evaluation/
├── Figures/ # Plots and visualizations
└── README.md # This file

---

## 🔄 Reproducibility

We ensure reproducibility through:

- ✅ Fixed random seeds
- ✅ Language-wise stratified splits
- ✅ Standard evaluation metrics (Accuracy, Precision, Recall, F1)

---

## 📝 Citation

If you use the **IndicHM Dataset** in your research, please cite:

```bibtex
@article{indichm2024,
  title={IndicHM: A Large-Scale Multilingual Multimodal Benchmark for Hateful Meme Detection},
  author={[Author Names]},
  journal={[Journal/Conference]},
  year={2024}
}
