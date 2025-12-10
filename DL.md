# Architectures de Deep Learning — Guide simplifié

## Sommaire  
1. [Réseaux de neurones artificiels (ANN / MLP)](#ann)  
2. [Réseaux convolutifs (CNN)](#cnn)  
3. [Réseaux récurrents (RNN)](#rnn)  
4. [Long Short-Term Memory (LSTM)](#lstm)  
5. [Gated Recurrent Unit (GRU)](#gru)  
6. [Graph Neural Network (GNN)](#gnn)  
7. [Notes pratiques & quand utiliser quoi](#pratique)  

---

<a name="ann"></a>
## 1. Réseaux de neurones artificiels (ANN / MLP)  
**Principe**  
Un ANN (ou MLP — multilayer perceptron) est un réseau à propagation avant (feed-forward) composé de plusieurs couches de neurones, chaque neurone étant connecté à tous les neurones de la couche précédente. :contentReference[oaicite:3]{index=3}  

**Fonctionnement (à une couche cachée)**  
h = activation(W1 · x + b1)
ŷ = output_fn(W2 · h + b2)

où `activation` peut être ReLU, tanh, etc.  

**Usage**  
- Données tabulaires, features non structurées.  
- Problèmes simples de classification ou régression.  

**Limites**  
- Inefficace sur des données structurées (images, séquences) : beaucoup de paramètres, aucune prise en compte de la structure spatiale ou temporelle.  

---

<a name="cnn"></a>
## 2. Réseaux convolutifs (CNN)  
**Principe**  
Les CNN exploitent la structure spatiale des données (comme les images) via des **filtres (kernels)** partagés et des opérations de convolution — au lieu de connexions denses. Cela réduit fortement le nombre de paramètres et capture les **caractéristiques locales** (bordures, textures…) progressivement jusqu’à des structures complexes. :contentReference[oaicite:4]{index=4}  

**Structure typique**  
- Couche convolution + activation (ex. ReLU)  
- (Optionnel) Pooling / sous-échantillonnage — pour réduire dimension et abstraction  
- Plusieurs blocs convolution/pooling  
- Couche(s) fully-connected finale(s) pour classification ou sortie  

**Quand l’utiliser**  
- Vision par ordinateur : classification d’images, détection d’objets, segmentation, etc.  
- Données avec structure spatiale (images, audio spectrogrammes, parfois textes via embeddings).  

**Avantages**  
- Moins de paramètres qu’un ANN dense équivalent.  
- Capable d’extraire des motifs invariants (translation, motifs répétés).  

---

<a name="rnn"></a>
## 3. Réseaux récurrents (RNN)  
**Principe**  
Un RNN traite des **séquences** (texte, audio, séries temporelles…) en maintenant un **état interne (mémoire)**. À chaque pas de temps, la sortie dépend de l’entrée actuelle **et** de l’état précédemment mémorisé. :contentReference[oaicite:5]{index=5}  

**Formule simple**  
h_t = φ(W_x · x_t + W_h · h_{t-1} + b)
y_t = output_fn(h_t)


**Applications**  
- Traitement automatique du langage (NLP), traduction, reconnaissance vocale.  
- Séries temporelles, prévisions, génération de texte, etc.  

**Limites**  
- Problèmes de **vanishing / exploding gradient** quand la séquence est longue — l’information du passé se perd ou devient instable pendant l’entraînement. :contentReference[oaicite:6]{index=6}  
- Entraînement séquentiel (moins parallélisable), plus lent.  

---

<a name="lstm"></a>
## 4. Long Short-Term Memory (LSTM)  
**Pourquoi LSTM ?**  
Pour pallier le problème de disparition du gradient des RNN et permettre la mémorisation d’informations importantes sur de longues séquences. :contentReference[oaicite:7]{index=7}  

**Mécanisme clé : portes + cellule de mémoire**  
- Un état de cellule `c_t` conserve l’information long terme.  
- Trois « portes » contrôlent le flux de l’information :  
  - *forget gate* : décider ce qu’on oublie  
  - *input gate* : décider ce qu’on ajoute  
  - *output gate* : décider ce qu’on lit  

→ Cette organisation permet de garder des informations pertinentes sur de longues périodes. :contentReference[oaicite:8]{index=8}  

**Usage typique**  
- Traduction, modélisation de langage, reconnaissance vocale, séries temporelles longues, etc.  

---

<a name="gru"></a>
## 5. Gated Recurrent Unit (GRU)  
**Principe**  
Une variante plus simple des LSTM : moins de portes, seulement un seul état caché (pas d’état cellule séparé), ce qui rend le modèle plus léger et plus rapide à entraîner, tout en gardant la capacité à gérer des dépendances dans le temps. :contentReference[oaicite:9]{index=9}  

**Gates principales**  
- *update gate* (miroir des input + forget gates)  
- *reset gate*  

**Quand l’utiliser**  
- Quand on a des contraintes de ressources (calcule, mémoire), ou des séquences moyennes.  
- Souvent comparable aux LSTM en performance pour de nombreuses tâches.  

---

<a name="gnn"></a>
## 6. Graph Neural Network (GNN)  
**Contexte / problème**  
Quand les données sont structurées en **graphes** — nœuds + arêtes + (optionnellement) attributs — par exemple réseaux sociaux, molécules, knowledge-graphs, etc.  

**Principe général**  
Chaque nœud met à jour sa représentation en agrégeant l’information des nœuds voisins (et éventuellement des arêtes). On fait ça sur plusieurs « couches », ce qui permet de diffuser l’information dans le graphe.  

**Forme typique** (simplifiée) :  
h_v^{(l+1)} = UPDATE( h_v^{(l)}, AGGREGATE( { h_u^{(l)} | u ∈ N(v) } ) )
où `h_v^{(l)}` est la représentation du nœud v à la couche l, et `N(v)` l’ensemble de ses voisins.  

**Applications**  
- Classification de nœuds (ex : prédire un label pour chaque entité), prédiction de liens, classification de graphes complets (ex : molécules), recommandation, etc.  

**Pourquoi utile**  
- Permet de traiter des données non structurées « spatiales » ou relationnelles, impossible avec CNN ou RNN classiques.  

---

<a name="pratique"></a>
## 7. Notes pratiques & guide de choix  

| Type de donnée / problème | Architecture recommandée |
|--------------------------|--------------------------|
| Données tabulaires, non structurées | ANN / MLP |
| Images, signaux avec structure spatiale | CNN |
| Séquences (texte, audio, séries temporelles) courtes ou moyennes | RNN, GRU, LSTM |
| Séquences longues avec dépendances lointaines | LSTM (ou GRU) |
| Données en forme de graphe (relations, réseaux, molécules…) | GNN |
| Critères : mémoire, capacité, rapidité | GRU — souvent bon compromis |
| Critères : capacité maximale, dépendances longues | LSTM |

**Conseils d’usage**  
- Commencez simple : MLP (ou CNN/RNN selon la structure) puis complexifiez si nécessaire.  
- Pour les données structurées (images, graphes…) — préférez des architectures adaptées (CNN, GNN).  
- Pour séquences longues : préférez des architectures avec contrôle de la mémoire (LSTM / GRU).  
- Testez différentes architectures — souvent la meilleure dépend du dataset et de la tâche.  

---

## 8. Références pour approfondir  
- Multilayer Perceptron (MLP) — définition et fonctionnement de base. :contentReference[oaicite:10]{index=10}  
- CNN — convolution, pooling, structure, avantages pour l’image. :contentReference[oaicite:11]{index=11}  
- RNN — principes, mémoire interne, séquences. :contentReference[oaicite:12]{index=12}  
- LSTM — traitement des longues dépendances, mémorisation. :contentReference[oaicite:13]{index=13}  
- GRU — variante légère des RNN avec gates simplifiées. :contentReference[oaicite:14]{index=14}  
- GNN — traitement de données en graphe (concept général). (Pour le principe générique, littérature spécialisée recommandée)  


