#AI-Based Biodiversity Detection from eDNA
Overview

This project focuses on building an intelligent system that can analyze environmental DNA (eDNA) sequences and identify the organisms present in a sample. The system is designed to work even when traditional biological databases fail to recognize unknown species.

By combining machine learning and clustering techniques, the model can classify known organisms, detect unknown sequences, and estimate biodiversity in a given sample.


#Problem Statement

Understanding biodiversity in deep-sea ecosystems is a major challenge. Scientists collect water or sediment samples and extract DNA fragments from them. These fragments represent organisms living in that environment.

Traditional methods rely on comparing DNA sequences with existing databases. However, these approaches face several limitations:

Many deep-sea organisms are not present in reference databases
Sequence matching is slow and computationally expensive
Unknown species cannot be identified or analyzed properly

As a result, a large portion of biodiversity remains undiscovered or incorrectly classified.

#Proposed Solution

This project introduces an AI-based pipeline that learns patterns directly from DNA sequences instead of depending entirely on databases.

The system performs the following tasks:

Classifies DNA sequences into taxonomic groups (phylum level)
Detects sequences that do not match known patterns
Groups unknown sequences into clusters representing potential new species
Calculates biodiversity distribution within the sample

This approach allows faster analysis and enables the discovery of novel organisms.


