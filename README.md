# IndicHM: Indic Hateful Meme Dataset

**IndicHM** is a large-scale multilingual and multimodal benchmark for hateful meme detection in **Hindi, English, and Bengali**. The dataset contains **15,004 memes** annotated for binary hate detection and fine-grained hate-target classification.

The benchmark is designed to support research on **multilingual multimodal hate speech detection**, with particular emphasis on Indic languages, code-mixed content, and the interaction between visual and textual signals.

> **⚠️ Dataset access is restricted to academic and research purposes.**
>
> The dataset is **not publicly downloadable**. Researchers must complete the consent and access-request process described below. Redistribution of the dataset is strictly prohibited.

---

## 📌 Overview

IndicHM contains **15,004 hateful and non-hateful memes** distributed across three languages:

* 🇮🇳 **Hindi**
* 🇬🇧 **English**
* 🇧🇩 **Bengali**

Hateful memes are further annotated into five target categories:

1. **Gender-based Hate**
2. **Religious Hate**
3. **Political Hate**
4. **Racist Hate**
5. **Other Forms of Hate**

The Hindi and Bengali subsets additionally contain **code-mixed and mixed-script content**, reflecting common patterns in multilingual social-media communication.

---

## 🎯 Motivation

Memes frequently communicate harmful or hateful messages through the interaction of **textual and visual information**. A meme may appear harmless when either modality is considered independently, while the combination of both can convey a hateful meaning.

Existing hateful-meme benchmarks are predominantly English-centric, leaving multilingual and code-mixed settings comparatively underrepresented. This is particularly important for India and South Asia, where social-media communication commonly involves multiple languages, scripts, transliteration, and code-mixing.

IndicHM addresses this gap by providing a dedicated multilingual multimodal benchmark for studying hateful content in Indic-language settings.

---

# 📊 Dataset

## Languages and Categories

| Language  |      Hate |  Non-Hate |    Gender |  Religion | Political |    Racism |    Others |      Total |
| :-------- | --------: | --------: | --------: | --------: | --------: | --------: | --------: | ---------: |
| Hindi     |     2,509 |     2,500 |       680 |       608 |       560 |       223 |       438 |      5,009 |
| English   |     2,495 |     2,500 |       810 |       695 |       409 |       332 |       249 |      4,995 |
| Bengali   |     2,500 |     2,500 |       500 |       500 |       500 |       500 |       500 |      5,000 |
| **Total** | **7,504** | **7,500** | **1,990** | **1,803** | **1,469** | **1,055** | **1,187** | **15,004** |

### Dataset Highlights

* **15,004** total multimodal memes
* **7,504** hateful memes
* **7,500** non-hateful memes
* **3 languages**
* **5 hate-target categories**
* Multilingual and **code-mixed** content
* Balanced language distribution
* Language-wise stratified evaluation splits

---

# 🧪 Evaluation Tasks

## Task 1 — Binary Hate Detection

Given a meme containing an image and associated text, the model predicts whether the meme is:

* **Hate**
* **Non-Hate**

### Split

| Split      |    Samples | Percentage |
| :--------- | ---------: | ---------: |
| Train      |     12,003 |     80.00% |
| Validation |      1,500 |     10.00% |
| Test       |      1,501 |     10.00% |
| **Total**  | **15,004** |   **100%** |

### Label Distribution

| Set         |           Non-Hate |               Hate |
| :---------- | -----------------: | -----------------: |
| Train       |     6,000 (49.99%) |     6,003 (50.01%) |
| Validation  |       750 (50.00%) |       750 (50.00%) |
| Test        |       750 (49.97%) |       751 (50.03%) |
| **Overall** | **7,500 (49.99%)** | **7,504 (50.01%)** |

---

## Task 2 — Hate Target Classification

For memes identified as hateful, the model predicts the **dominant hate-target category**:

* Gender-based Hate
* Religious Hate
* Political Hate
* Racist Hate
* Other Forms of Hate

### Split

| Split      |   Samples | Percentage |
| :--------- | --------: | ---------: |
| Train      |     6,001 |     79.97% |
| Validation |       750 |      9.99% |
| Test       |       753 |     10.03% |
| **Total**  | **7,504** |   **100%** |

### Category Distribution

| Category     |     Train | Validation |    Test |   Overall |
| :----------- | --------: | ---------: | ------: | --------: |
| Gender-based |     1,592 |        199 |     199 |     1,990 |
| Others       |       949 |        119 |     119 |     1,187 |
| Political    |     1,175 |        147 |     147 |     1,469 |
| Racism       |       843 |        105 |     107 |     1,055 |
| Religion     |     1,442 |        180 |     181 |     1,803 |
| **Total**    | **6,001** |    **750** | **753** | **7,504** |

---

# 🌐 Language Distribution

IndicHM maintains an approximately balanced distribution across the three languages.

### Task 1

| Split      |          Hindi |        English |        Bengali |
| :--------- | -------------: | -------------: | -------------: |
| Train      | 4,007 (33.38%) | 3,996 (33.29%) | 4,000 (33.33%) |
| Validation |   501 (33.40%) |   499 (33.27%) |   500 (33.33%) |
| Test       |   501 (33.38%) |   500 (33.31%) |   500 (33.31%) |

### Task 2

| Split      |          Hindi |        English |        Bengali |
| :--------- | -------------: | -------------: | -------------: |
| Train      | 2,006 (33.43%) | 1,995 (33.24%) | 2,000 (33.33%) |
| Validation |   251 (33.47%) |   249 (33.20%) |   250 (33.33%) |
| Test       |   252 (33.47%) |   251 (33.33%) |   250 (33.20%) |

---

# 🔤 Code-Mixed and Script Statistics

IndicHM includes multilingual and mixed-script patterns commonly observed in online communication.

| Language  |      Total | Code-Mixed | Code-Mixed % | Script-Switched | Transliterated |
| :-------- | ---------: | ---------: | -----------: | --------------: | -------------: |
| Hindi     |      5,009 |      2,055 |       41.03% |             582 |          1,473 |
| Bengali   |      5,000 |      1,105 |       21.10% |           1,046 |             59 |
| English   |      4,995 |          — |            — |               — |              — |
| **Total** | **15,004** |  **3,160** |  **~21.06%** |       **1,628** |      **1,532** |

> **Note:** Code-mixed statistics are reported for Hindi and Bengali, where mixed-language content is present. English memes are treated as monolingual English samples.

---

# 🧠 IndicHMNet

Along with the dataset, we introduce **IndicHMNet**, a task-aware multilingual multimodal framework for hateful meme detection.

IndicHMNet jointly models visual and textual information while explicitly addressing **cross-modal disagreement/incongruity**.

### Architecture

The framework consists of the following key components:

**1. Multimodal Representation**

A pretrained **SigLIP2** encoder extracts aligned image–text representations.

**2. Disagreement-Aware Fusion**

Visual and textual representations are combined using a gated fusion mechanism that accounts for disagreement between the modalities and adaptively emphasizes the more informative modality.

**3. Hate Detection**

Task 1 is optimized using supervised binary classification together with **label-conditioned retrieval-guided contrastive learning**.

**4. Target Classification**

Task 2 is trained independently using **cross-entropy loss** over the five hate-target categories.

---

# 🔬 Baseline Models

We evaluate IndicHM using representative unimodal and multimodal architectures.

### Text Encoders

* **MuRIL**
* **mBERT**
* **XLM-R**

### Vision Encoders

* **ViT**
* **ConvNeXt**

### Multimodal Framework

* **IndicHMNet**

  * SigLIP2-based multimodal representation
  * Disagreement-aware gated fusion
  * Retrieval-guided contrastive learning
  * Independent target classification

---

# 📈 Evaluation Metrics

We use standard classification metrics:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**

For reproducibility, experiments use:

* Fixed random seeds
* Language-wise stratified splits
* Consistent train/validation/test partitions
* Standard evaluation protocols

---

# 📁 Repository Structure

```text
Indic-Hateful-Meme-Dataset/
│
├── Images/
│   └── Dataset images
│
├── Code/
│   ├── Model implementations
│   ├── Training scripts
│   └── Evaluation scripts
│
├── Figures/
│   └── Plots and visualizations
│
├── README.md
└── LICENSE
```

> **Important:** The repository does **not** redistribute the IndicHM dataset. Dataset files must be obtained through the official access procedure.

---

# 🔐 Dataset Access

Access to IndicHM is restricted to **academic and non-commercial research**.

Before requesting access, please ensure that you agree to the following conditions.

### 1. Non-Commercial Use

The dataset may only be used for legitimate academic and research purposes.

Commercial use, either directly or indirectly, is prohibited. The dataset must not be used to develop, train, evaluate, or market commercial products or services.

### 2. No Redistribution

The dataset, in whole or in part, must **not** be:

* Redistributed
* Republished
* Uploaded to public repositories
* Uploaded to private repositories accessible to unauthorized users
* Shared through social media
* Shared through cloud storage or other platforms with unauthorized individuals

Researchers are responsible for storing the dataset securely.

### 3. Platform Policy Compliance

IndicHM is derived from publicly available data originating from **Pinterest and Reddit**.

Users are responsible for complying with the applicable platform policies, developer agreements, data-use requirements, privacy requirements, and redistribution restrictions.

### 4. Academic Integrity and Responsible Use

The dataset must only be used for legitimate academic research. Researchers are responsible for ensuring that their use of the dataset follows applicable ethical standards, institutional requirements, and responsible research practices.

---

# 📝 How to Request Dataset Access

**Please use your institutional email address.**

Requests submitted using personal email addresses will **not be considered**.

### Step 1 — Complete the Consent Form

First, complete the required:

**[IndicHM Dataset Consent Form](https://docs.google.com/document/d/1G1kSiTy53pZ5KZOVCoTQCKTJ415cYCTL/edit?usp=sharing&ouid=107493269096531134179&rtpof=true&sd=true)**

### Step 2 — Submit the Dataset Request

Upload the completed consent form through the:

**[IndicHM Dataset Request Form](https://forms.gle/CLT4pM4NTKZYWCEG7)**

You will be asked to provide your academic/institutional details.

### Step 3 — Review and Approval

Each request will be reviewed to verify compliance with the dataset access requirements.

Once your request has been **formally reviewed and approved**, you will receive an email containing instructions for accessing and downloading the dataset.

> **Dataset files are not provided through this GitHub repository.**

---

# ⚖️ Terms of Use

By requesting or using IndicHM, researchers agree that they will:

* Use the dataset exclusively for academic and non-commercial research.
* Not redistribute or republish the dataset.
* Not make the dataset publicly accessible.
* Follow applicable platform policies and data-use restrictions.
* Maintain appropriate security controls for dataset storage.
* Use the dataset responsibly and in accordance with institutional research requirements.

Access may be denied or revoked if these conditions are violated.

---

# 📚 Citation

If you use **IndicHM** or **IndicHMNet** in your research, please cite the associated publication:

```bibtex
@inproceedings{indicHM,
  title     = {IndicHM: A Multilingual Multimodal Hateful Meme Dataset},
  author    = {Shahid Shafi Dar and Mohd Aamir and Arindol Sarkar and Hrishiraj Chowdhury},
  booktitle = {Proceedings of the ...},
  year      = {2026}
}
```

> Please replace the BibTeX entry with the final publication metadata once the paper is officially published.

---

# 🤝 Responsible Research

IndicHM is intended to facilitate research on multilingual hateful-content detection and **AI for social good**.

Researchers are encouraged to consider the potential societal impact of automated hate detection systems, including false positives, false negatives, language-specific performance disparities, and the challenges associated with culturally and linguistically diverse online content.

---

# 📬 Contact

For questions regarding dataset access, research collaboration, or the IndicHM benchmark, please contact the authors through the contact information provided with the associated publication.

---

## ⭐ Acknowledgement

We thank researchers and institutions interested in advancing **multilingual, multimodal, and socially responsible AI** for their interest in IndicHM.
