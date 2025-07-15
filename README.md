# S-HARM

Repository for the "'Humor, Art, or Misinformation?': A Multimodal Dataset for Intent-Aware Synthetic Image Detection"

## Abstract
Recent advances in multimodal AI have enabled progress in detecting synthetic and out-of-context content. However, existing work largely overlooks the intent behind AI-generated images. To fill this gap, we introduce \textit{S-HARM}, a multimodal dataset for intent-aware classification, comprising 9,576 'in the wild' image–text pairs from Twitter/X and Reddit, labeled as \textit{Humor/Satire}, \textit{Art}, or \textit{Misinformation}. Additionally, we explore three prompting strategies (image-guided, description-guided, and multimodally-guided) to construct a large-scale synthetic training dataset with Stable Diffusion. We conduct an extensive comparative study including modality fusion, contrastive learning, reconstruction networks, attention mechanisms, and large vision-language models. Our results show that models trained on image- and multimodally-guided data generalize better to 'in the wild' content, due to preserved visual context. However, overall performance remains limited, underscoring the complexity of inferring intent and the need for specialized architectures.

![Screenshot](banner.pdf)

## Preparation
- Clone this repo:
```
git clone https://github.com/Qedrigord/SHARM
cd SHARM
```

- Create a python enviroment

- Install all dependencies with: `pip install -r requirements.txt`


## Contents
- scraping: Scripts used to mine and filter data from Twitter and Reddit

- generation: Contains the code that was used for the generation of the 3 different train sets

- train data: Image and text feature representations for the training sets

- test data: Feature representations for test samples

- experiments: All training and evaluation code for the conducted experiments

## Acknowledgements
This work is partially funded by the ``DisAI'' project under Grant aggrement 101079164. 
The authors would also like to acknowledge the support and computational resources provided by the IT Center of the Aristotle University of Thessaloniki (AUTh) throughout the progress of this research work.

## Licence

## Contact
Stefanos-Iordanis Papadopoulos (stefpapad@iti.gr)