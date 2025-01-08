# NHITS Project

Ce dépôt présente notre implémentation de l'algorithme NHITS. Vous trouverez ici les instructions nécessaires pour télécharger les datasets et lancer des entraînements.

## Pré-requis

Avant de commencer, assurez-vous d'avoir installé **Python**  (version recommandée : 3.7 ou supérieure).


---

## Étapes pour utiliser le projet

### 1. Téléchargement des datasets

Pour télécharger les datasets nécessaires, exécutez le script `get_dataset.py`. Ce script créera un dossier `datasets` contenant les fichiers suivants :

- `ECL`
- `ETTh1`
- `ETTh2`
- `ETTm1`
- `ETTm2`
- `Exchange`
- `ili`
- `traffic`
- `weather`

**Commande à exécuter** dans un terminal :

```bash
python get_dataset.py
```
 
### 2. Lancement d'un entraînement

Pour entraîner le modèle sur un dataset avec un horizon spécifique, utilisez le script `multivariate_training.py` situé dans le répertoire `src`.

#### Via un terminal

Placez-vous dans le répertoire `src` et exécutez la commande suivante :

```bash
python3 multivariate_training.py --dataset "ETTm2" --horizon 96
```
ou si votre commande Python est différente :
```bash
python  multivariate_training.py --dataset "ETTm2" --horizon 96
```
- --dataset : Nom du dataset à utiliser (exemple : "ETTm2"). 
- --horizon : Longueur de l'horizon (exemple : 96).

#### Via un éditeur de code

Ouvrez le fichier `multivariate_training.py` dans votre éditeur de code et modifiez les lignes suivantes :

- `Ligne 13` : Définissez le nom du dataset.

- `Ligne 14` : Définissez l'horizon.

Ensuite, exécutez le fichier directement depuis l'éditeur.

Cela va permettre d'enregistrer dans le repértoire `results`, les hyperparamètres les plus optimales pour le modèle selon le dataset et l'horizon.

`Attention` : Certaines valeurs d’horizon peuvent être trop grandes pour certains datasets, ce qui entraînera un message d'erreur. Adaptez les paramètres en fonction de vos besoins et des limitations du dataset.

### Evaluation du modèle
Pour évaluer le modèle selon un dataset et un horizon, utilisez le script `evaluate_model.py` situé dans le répertoire `src`.

#### Via un terminal

Placez-vous dans le répertoire `src` et exécutez la commande suivante :

```bash
python3 evaluate_model.py --dataset "ETTm2" --horizon 96
```
ou si votre commande Python est différente :
```bash
python  evaluate_model.py --dataset "ETTm2" --horizon 96
```
- --dataset : Nom du dataset à utiliser (exemple : "ETTm2"). 
- --horizon : Longueur de l'horizon (exemple : 96).

#### Via un éditeur de code

Ouvrez le fichier `evaluate_model.py` dans votre éditeur de code et modifiez les lignes suivantes :

- `Ligne 17` : Définissez le nom du dataset.

- `Ligne 18` : Définissez l'horizon.

Ensuite, exécutez le fichier directement depuis l'éditeur.

Le code va afficher la MAE et la MSE du modèle.

---

**Note** : le dossier `archives` contient des codes que nous avons implémenté durant le projet, mais qu'ils ne sont plus utiles à présent.
