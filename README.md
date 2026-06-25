\# 🔭 EO Change Detection — Semantic Change Detection for Satellite Imagery



Détection de changement sémantique sur images Sentinel-2 en utilisant 

les embeddings RemoteCLIP et la similarité cosinus.



!\[Demo](assets/demo.gif)



\---



\## 🎯 Objectif



Explorer l'utilisation des \*\*embeddings de modèles de fondation géospatiaux\*\* 

pour détecter et caractériser des changements temporels sur des images 

Earth Observation (EO), dans le cadre d'une candidature à l'alternance 

Data Scientist chez CLS.



\---



\## 💡 Principe

Image Avant (date 1)  →  RemoteCLIP  →  Embedding A (512 dims)



Image Après (date 2)  →  RemoteCLIP  →  Embedding B (512 dims)



↓



Score = 1 - cosine\_similarity(A, B)



↓



Vecteur différence → Type de changement



\---



\## 📊 Résultats



| Ville | Date avant | Date après | Score | Interprétation |

|-------|-----------|-----------|-------|----------------|

| Pisa | 20150704 | 20180211 | 0.0847 | Changement modéré |

| Bordeaux | 20160504 | 20171026 | 0.0733 | Changement modéré |

| Nantes | 20150821 | 20171014 | 0.0714 | Changement modéré |

| Paris | 20161130 | 20171107 | 0.0453 | Stable |

| Rennes | 20150821 | 20170621 | 0.0358 | Stable |



\*\*Seuils de détection :\*\*

\- Score < 0.05 → Stable

\- Score 0.05–0.15 → Changement modéré  

\- Score > 0.15 → Changement majeur



\---



\## 🏷️ Types de changement détectés



Détection basée sur la similarité entre le vecteur de différence 

`(emb\_after - emb\_before)` et des descriptions textuelles prédéfinies :



\- Urban expansion and new construction

\- Deforestation and loss of vegetation

\- Agricultural field changes and crop rotation

\- Flooding and water level changes

\- Industrial development

\- Seasonal vegetation changes



\---



\## 🛠️ Stack technique



\- \*\*Modèle\*\* : RemoteCLIP (flax-community/clip-rsicd-v2)

\- \*\*Métrique\*\* : Similarité cosinus sur embeddings 512D

\- \*\*Dataset\*\* : OSCD (Onera Satellite Change Detection) — 24 villes mondiales

\- \*\*Interface\*\* : Streamlit

\- \*\*Traitement EO\*\* : rasterio, numpy



\---



\## 📁 Structure du projet



\---



\## 🚀 Installation



```bash

git clone https://github.com/ton-username/eo-change-detection.git

cd eo-change-detection

pip install -r requirements.txt

streamlit run app/streamlit\_app.py

```



\---



\## ⚠️ Limitations et perspectives



\*\*Limitations actuelles :\*\*

\- Scores faibles car RemoteCLIP encode le contenu global — 

&#x20; les changements locaux sont dilués dans l'embedding global

\- Pas de segmentation spatiale du changement



\*\*Perspectives :\*\*

\- Découpage en patches pour détecter les zones précises de changement

\- Fine-tuning sur données de changement annotées

\- Intégration d'un LLM vision pour description en langage naturel

\- Pipeline temporel sur données Sentinel-2 continues (GEE)



\---



\## 📚 Références



\- \[OSCD Dataset](https://rcdaudt.github.io/oscd/)

\- \[RemoteCLIP](https://huggingface.co/flax-community/clip-rsicd-v2)

\- \[Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)



\---



\*Projet 2/4 — Candidature alternance Data Scientist chez CLS (Ramonville-Saint-Agne)\*

