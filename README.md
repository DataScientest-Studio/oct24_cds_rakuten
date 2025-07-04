# Projet de Classification de Produits Rakuten

Ce projet a pour objectif de classifier des articles du site de Rakuten en 27 catégories distinctes. Pour ce faire, nous avons utilisé une approche multimodale, combinant l'analyse du texte descriptif et de l'image du produit. Le pipeline est divisé en plusieurs étapes, allant de la préparation des données à l'entraînement de modèles de Deep Learning.

Le projet a été réalisé par : Didier Ballouard, Frédérick de Miollis, Fahim Herriche et Bakari Traoré

---

## Fonctionnalités

* **Préparation complète des données** : Nettoyage de code HTML, détection de langue et traduction automatique en français.
* **Rééquilibrage des classes** : Utilisation de techniques de sur-échantillonnage et de sous-échantillonnage pour gérer le déséquilibre des catégories.
* **Modélisation Texte** : Entraînement et fusion de modèles Transformers (CamemBERT, Flaubert).
* **Modélisation Image** : Entraînement et fusion de plusieurs architectures de Computer Vision (Maxvit, ConvNeXtV2, EfficientNetV2, etc.).
* **Méta-Apprentissage** : Création d'un méta-modèle final qui combine les prédictions des modèles texte et image pour une classification multimodale robuste.
* **Optimisation** : Recherche d'hyperparamètres avec Optuna pour optimiser la performance des méta-modèles.

---

## Structure du Projet

Le projet est organisé en quatre notebooks Jupyter, chacun représentant une étape clé du pipeline :

1.  **`1.0_Projet_Rakuten_Preparartion_des_donnees.ipynb`**
    * Ce notebook est le point de départ. Il fusionne les données textuelles, nettoie le contenu (suppression des balises HTML notamment), détecte la langue de chaque texte et traduit les textes non-francophones en français à l'aide de `deep_translator`. Enfin, il rééquilibre les classes et sauvegarde un jeu de données final (`X_train_fr_final.csv`) ainsi que l'encodeur de labels (`label_encoder_final.pkl`).

2.  **`2.0_Projet_Rakuten_Deep_Learning_Texte.ipynb`**
    * Ce notebook se concentre sur la classification textuelle.
    * **Partie 1** : Entraînement de modèles de base (CamemBERT, Flaubert) sur les descriptions de produits.
    * **Partie 2** : Fusion (stacking) des prédictions (logits) de ces modèles pour créer un méta-modèle texte optimisé avec LightGBM et un MLP.

3.  **`3.0_Projet_Rakuten_Deep_Learning_Images.ipynb`**
    * Ce notebook gère la classification des images.
    * **Partie 1** : Décompression des images et entraînement par lots de plusieurs modèles de Computer Vision (Maxvit, ConvNeXtV2, EfficientNetV2, CoAtNet, GCViT) à l'aide de la bibliothèque `timm`.
    * **Partie 2** : Fusion et stacking des prédictions de ces modèles pour créer un méta-modèle image optimisé.

4.  **`4.0_Projet_Rakuten_Deep_Learning_Meta.ipynb`**
    * C'est le notebook final qui assemble toutes les prédictions.
    * **Partie 1** : Recherche de la meilleure combinaison de modèles texte et image pour créer un fichier de prédictions "booster".
    * **Partie 2** : Entraînement d'un méta-modèle global qui prend en entrée les prédictions de **tous** les modèles individuels (texte et image) pour fournir la classification finale.

---

## Installation

Avant de lancer les notebooks, il est nécessaire d'installer les dépendances. Chaque notebook contient les commandes d'installation requises. Les principales bibliothèques utilisées sont :

```bash
pip install pandas numpy torch torchvision timm
pip install joblib scikit-learn matplotlib seaborn
pip install transformers sacremoses
pip install optuna lightgbm
pip install googletrans==4.0.0-rc1 deep_translator beautifulsoup4
pip install fasttext
```

Les notebooks génèrent également des fichiers `requirements_*.txt` qui peuvent être utilisés pour recréer l'environnement :

```bash
pip install -r requirements_Data.txt
pip install -r requirements_Texte.txt
pip install -r requirements_Images.txt
pip install -r requirements_Meta.txt
```

-----

## Déroulement du Pipeline

Pour reproduire les résultats, veuillez exécuter les notebooks dans l'ordre suivant :

### **Étape 1 : Préparation des Données**

  * Exécutez `1.0_Projet_Rakuten_Preparartion_des_donnees.ipynb`.
  * Assurez-vous que les données brutes (`X_train.csv`, `Y_train.csv`) et le modèle `lid.176.bin` (pour fasttext) sont accessibles.
  * Ce script générera les fichiers `X_train_fr_final.csv`, `label_encoder_final.pkl` et `label_mapping_final.json`.

### **Étape 2 : Entraînement des Modèles Texte**

  * Exécutez `2.0_Projet_Rakuten_Deep_Learning_Texte.ipynb`.
  * Ce notebook utilise `X_train_fr_final.csv` pour entraîner les modèles texte.
  * Il sauvegardera les prédictions (logits) de chaque modèle (ex: `camembert2_logits.pt`, `flaubert2_logits.pt`) ainsi que les méta-modèles texte.

### **Étape 3 : Entraînement des Modèles Image**

  * Exécutez `3.0_Projet_Rakuten_Deep_Learning_Images.ipynb`.
  * Assurez-vous que le fichier `image_train.zip` est accessible. Le script le décompressera automatiquement.
  * Il entraînera les modèles de Computer Vision et sauvegardera leurs prédictions (ex: `maxvit_base_tf_224_logits.pt`, `convnextv2_base_logits.pt`).

### **Étape 4 : Entraînement du Méta-Modèle Final**

  * Exécutez `4.0_Projet_Rakuten_Deep_Learning_Meta.ipynb`.
  * Ce notebook chargera les prédictions de tous les modèles précédents (texte et image) pour entraîner le classifieur multimodal final.
  * L'artefact final, `final_meta_model_lgbm.joblib` (ou `.pkl`), sera alors sauvegardé. C'est ce modèle qui doit être utilisé pour les prédictions finales.

-----

## Modèles Utilisés

### **Modèles de base (Texte)**

  * CamemBERT
  * FlauBERT

### **Modèles de base (Image)**

  * MaxViT
  * ConvNeXt V2
  * EfficientNet V2
  * CoAtNet
  * GCViT

### **Méta-Modèles (Ensembling & Stacking)**

  * LightGBM
  * Perceptron Multi-Couches (MLP)
  * Régression Logistique

---

## Organisation du projet

├── README.md          <- README principal pour les développeurs utilisant ce projet.
├── data
│   ├── processed      <- Jeux de données finaux et canoniques pour la modélisation.
│   └── raw            <- Dépôt des données brutes
│
├── models             <- Modèles entraînés et logits
│
├── notebooks          <- Dossier pour les Jupyter notebooks.
│   └── main           <- Notebooks en version finale.
│   └── versions       <- Versions de travail ou plus anciennes.
│
├── reports            <- Rapport du projet
│
├── requirements       <- Dossier des fichiers de dépendances.
│   └── requirements_Data.txt      <- Le fichier de dépendances pour reproduire l'environnement d'analyse du notebook `1.0_Projet_Rakuten_Preparartion_des_donnees.ipynb`.
│   └── requirements_Texte.txt     <- Le fichier de dépendances pour reproduire l'environnement d'analyse du notebook `2.0_Projet_Rakuten_Deep_Learning_Texte.ipynb`.
│   └── requirements_Images.txt    <- Le fichier de dépendances pour reproduire l'environnement d'analyse du notebook `3.0_Projet_Rakuten_Deep_Learning_Images.ipynb`.
│   └── requirements_Meta.txt      <- Le fichier de dépendances pour reproduire l'environnement d'analyse du notebook `4.0_Projet_Rakuten_Deep_Learning_Meta.ipynb`.

<!-- end list -->