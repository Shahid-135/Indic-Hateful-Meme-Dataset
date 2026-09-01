# Dataset Access
Access to the dataset is restricted to academic and research purposes. To request access, please ensure compliance with the following requirements:

- **Non-Commercial Use:** The dataset is provided exclusively for academic research and non-commercial purposes. Any form of commercial use, direct or indirect, is strictly prohibited. It must not be used to develop or market commercial products or services.

- **No Redistribution:** The dataset must not be redistributed, republished, or shared in any form, whether in full or in part, across any public or private repositories, platforms, or social media. Users must store the dataset in secure environments and ensure that unauthorized parties do not gain access.

- **Compliance with Platform Policies:** As this dataset is derived from publicly available Pinterest and Reddit data, users must comply with the platform's Developer Agreement and Policies. This includes adhering to data usage and privacy guidelines, ensuring that no actions breach content redistribution rules.

- **Academic Integrity:** The dataset must only be used for legitimate academic research. Researchers are responsible for ensuring that their use of the dataset does not contravene any ethical standards, particularly regarding data privacy and responsible usage.

1. Important: Use your institutional email address when submitting the form. Requests from personal email addresses will not be considered.
2. First, please fill out this [Consent Form](https://docs.google.com/document/d/1G1kSiTy53pZ5KZOVCoTQCKTJ415cYCTL/edit?usp=sharing&ouid=107493269096531134179&rtpof=true&sd=true), and then upload it in the [Dataset Request Form](https://forms.gle/CLT4pM4NTKZYWCEG7) along with your details. Once your request is reviewed and approved, you will receive an email with instructions on how to download the dataset.

# Overview
We introduce the IndicHM Dataset, a large-scale multilingual multimodal benchmark consisting of 15.004K hateful and non-hateful memes annotated across five hate categories—Gender-based Hate, Religious Hate, Political Hate, Racist Hate, and Other Forms of Hate. The dataset spans three languages: Hindi, English, and Bengali, with the Hindi and Bengali subsets containing code-mixed content that blends native scripts with English. Built upon this dataset, We propose IndicHMNet, a task-aware multilingual multimodal framework for hateful meme detection that jointly models visual and textual cues while explicitly addressing cross-modal incongruity. The model leverages a pretrained SigLIP2 encoder for aligned image–text representations and fuses them using a disagreement-aware gated mechanism to emphasize the dominant modality. Hate detection is trained with supervised classification augmented by label-conditioned retrieval-guided contrastive learning, while target classification is learned independently.

# Motivation
Hateful memes derive their meaning from the interaction between textual and visual cues, making them challenging for unimodal approaches. Despite their widespread use on social media, multilingual multimodal meme data—especially for Indic languages—remains scarce, with most existing resources being English-centric. This limitation is particularly critical in India, a linguistically diverse country with multiple regional languages, where online content is often multilingual and code-mixed. The absence of representative datasets for such settings motivates the creation of a dedicated multilingual multimodal hateful meme dataset (IndicHM) to support robust hate detection research in Indic languages.

# Dataset
Our dataset (IndicHM) is a multilingual multimodal resource for hate meme analysis spanning three languages: Hindi, English, and Bengali.
The dataset supports two evaluation tasks:

### Task–1: Binary Hate Detection
Classifies meme into Hate and Non-Hate.

### Task–2: Multiclass Hate Categorization
Categorizes hateful meme into one of five classes:
Gender, Religion, Political, Racism, and Others.

| Language | Hate | Non-Hate | Gender | Religion | Political | Racism | Others | Total |
|----------|-----:|---------:|-------:|---------:|----------:|--------:|-------:|------:|
| Hindi    | 2,509 | 2,500 | 680 | 608 | 560 | 223 | 438 | 5,009 |
| English  | 2,495 | 2,500 | 810 | 695 | 409 | 332 | 249 | 4,995 |
| Bengali  | 2,500 | 2,500 | 500 | 500 | 500 | 500 | 500 | 5,000 |
| **Total** | **7,504** | **7,500** | **1,990** | **1,803** | **1,469** | **1,055** | **1,187** | **15,004** |

## Task Description
Task–1 (Binary Classification):
Determines whether a given meme is Hateful or Non-Hateful.

Task–2 (Multiclass Classification):
For samples labeled as hate, the model predicts one of the following categories:

- Gender-based Hate  
- Religious Hate  
- Political Hate  
- Racist Hate  
- Other Forms of Hate
  
## Dataset Split and Distribution Statistics

### Task 1: Hate vs Non-Hate Classification

Split Size

| Split | Samples | Percentage |
|------|------:|------:|
| Train | 12,003 | 80.00% |
| Val   | 1,500 | 10.00% |
| Test  | 1,501 | 10.00% |
| **Total** | **15,004** | **100%** |

Label Distribution

| Set | Label 0 (Non-Hate) | Label 1 (Hate) |
|----|--------------------:|---------------:|
| Train | 6,000 (49.99%) | 6,003 (50.01%) |
| Val   | 750 (50.00%) | 750 (50.00%) |
| Test  | 750 (49.97%) | 751 (50.03%) |
| **Overall** | **7,500 (49.99%)** | **7,504 (50.01%)** |

Language-wise Distribution Across Splits

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

### Task 2: Hate Target Classification

Split Size

| Split | Samples | Percentage |
|------|------:|------:|
| Train | 6,001 | 79.97% |
| Val   | 750 | 9.99% |
| Test  | 753 | 10.03% |
| **Total** | **7,504** | **100%** |

Category Distribution

| Category | Train | Val | Test | Overall |
|--------|-----:|----:|----:|------:|
| Gender Based | 1,592 | 199 | 199 | 1,990 |
| Others       | 949 | 119 | 119 | 1,187 |
| Political    | 1,175 | 147 | 147 | 1,469 |
| Racism       | 843 | 105 | 107 | 1,055 |
| Religion     | 1,442 | 180 | 181 | 1,803 |

Language-wise Distribution Across Splits

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

## Mixed-code Statistics

| Language  | Total Samples | Code-Mixed | Code-Mixed % | Script-Switched | Transliterated |
| --------- | ------------: | ---------: | -----------: | --------------: | -------------: |
| Hindi     |         5,009 |      2,055 |       41.03% |             582 |          1,473 |
| Bengali   |         5,000 |      1,105 |       21.10% |             1046 |          59 |
| English   |         4,995 |          — |            — |               — |              — |
| **Total** |    **15,004** |  **3,160** |     **~21.06%** |       **1,628** |      **1,532** |


## Key Characteristics
Multilingual: Hindi, English, and Bengali

Balanced Dataset: Nearly equal distribution across languages and classes

Language Pattern: Hindi–English and Bengali–English code-mixed memes, as well as monolingual English memes.

## Models and Methodology
Text Models: MuRIL, mBERT, XLM-R  
Vision Models: ViT, ConvNeXt  

We propose a task-aware multilingual multimodal framework for hateful meme detection that jointly models visual and textual features while explicitly accounting for cross-modal incongruity. A pretrained SigLIP2 encoder extracts aligned image–text representations, which are fused using a disagreement-aware gated mechanism to adaptively emphasize the dominant modality. Hate detection is optimized using supervised classification augmented with label-conditioned retrieval-guided contrastive learning, while target classification is trained independently using cross-entropy loss.

## Folder Structure
Images/ – Dataset  
Code/ – Model, training, and evaluation code  
Figures/ – Plots and visualizations  

## Reproducibility
Fixed random seeds, language-wise stratified splits, and standard evaluation metrics (Accuracy, Precision, Recall, F1).

Once your request has been carefully reviewed and formally approved, you will receive a confirmation email containing detailed instructions on how to access and download the dataset.

Please note that full access and usage rights will be granted only after the approval process is completed, in accordance with our data usage policies.
