# Applications of Large Language Models in Italian Legal Document Analysis

## Overview
Large language models are utilized to analyze Italian legal documents primarily for tasks such as personal data de-identification, topic extraction, labeling, and summarization, as well as for benchmarking fairness across jurisdictions. While domain-specific models and pipelines achieve high accuracy in privacy and thematic analysis, evaluations in multilingual contexts reveal ongoing challenges with performance disparities.

## Key findings
Research demonstrates the effectiveness of tailored language models in processing sensitive Italian legal texts. For personal data de-identification, GiusBERTo serves as the first BERT-based model specialized for anonymizing Italian Court of Auditors decisions by recognizing entities like names, dates, and locations while maintaining contextual relevance [1]. Evaluated on a held-out test set, GiusBERTo achieves a 97% token-level accuracy in balancing privacy and data protection [1]. 

In broader document processing, pipelines integrating layout analysis, optical character recognition, and anonymization are used alongside large language models to analyze Italian Supreme Court judgments [2]. BERTopic extracts legal themes from these judgments, and large language models subsequently generate labels and summaries for the extracted topics [2]. When evaluated against domain expert interpretations, Claude Sonnet 3.7 achieves a BERTScore F1 of 0.8119 for labeling and 0.9130 for summarization [2]. 

Beyond specific domestic applications, language models are also studied using multilingual benchmarks. FairLex provides a benchmark suite that includes Italian datasets to evaluate the fairness of pre-trained language models and fine-tuning techniques across different fairness attributes [3]. 

## Open questions and gaps
Evaluations of language models on Italian and multilingual legal text processing highlight persistent performance group disparities [3]. Experiments indicate that none of the tested group-robust fine-tuning techniques consistently mitigate these disparities, pointing to open challenges in developing robust methods for legal natural language processing [3].

## References

[1] Giulio Salierno, Rosamaria Bertè, Luca Attias et al. (2024). *GiusBERTo: A Legal Language Model for Personal Data De-identification in Italian Court of Auditors Decisions*. http://arxiv.org/abs/2406.15032v1
[2] Matteo Marulli, Glauco Panattoni, Marco Bertini (2025). *A document processing pipeline for the construction of a dataset for topic modeling based on the judgments of the Italian Supreme Court*. http://arxiv.org/abs/2505.08439v1
[3] Ilias Chalkidis, Tommaso Pasini, Sheng Zhang et al. (2022). *FairLex: A Multilingual Benchmark for Evaluating Fairness in Legal Text Processing*. http://arxiv.org/abs/2203.07228v1
