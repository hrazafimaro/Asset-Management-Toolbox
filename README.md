# 📊 Asset Management Toolbox  
Outils d’analyse quanti et de modélisation financière (Python)

Ce projet regroupe plusieurs modules et notebooks permettant de reproduire des tâches courantes en **Asset Management**, **Gestion de Portefeuille**, **Risk Management** et **Pricing**.  
Il est conçu pour démontrer des compétences techniques en :

- Python
- Data Analysis
- Gestion d’actifs
- Optimisation de portefeuille
- Simulations Monte Carlo
- Pricing d’options
- Mesures de risque (VaR, CVaR)
- Backtesting de stratégies


---

## 🧠 Contenu pédagogique

### **1. Optimisation de portefeuille (Markowitz)**
- Calcul des rendements/volatilités
- Matrice de corrélation
- Frontière efficiente
- Portefeuille à volatilité minimale
- Maximisation du Sharpe Ratio

### **2. Backtesting de stratégies quantitatives**
- Simple Moving Average (SMA)
- Momentum
- Buy & Hold vs. stratégies dynamiques
- Mesures de performance :
  - CAGR
  - Max drawdown
  - Volatilité annualisée
  - Sharpe Ratio

### **3. Pricing par Monte Carlo**
- Pricing d’options européennes
- Modèle de Black–Scholes
- Génération de paths simulés
- Comparaison Monte Carlo vs. prix théorique

### **4. Gestion du risque (Risk Management)**
- Value at Risk (VaR)
- Conditional VaR (Expected Shortfall)
- VaR paramétrique, historique et Monte Carlo
- Distribution des pertes

---

## ▶️ Ouvrir les notebooks dans Google Colab

Vous pouvez exécuter les notebooks en un clic :

| Notebook | Lien |
|----------|------|
| Optimisation de portefeuille | [📘 Ouvrir dans Colab](https://colab.research.google.com/github/hrazafimaro/Asset-Management-Toolbox/blob/main/notebooks/01_Portfolio_Optimization.ipynb) |
| Backtesting | *(lien à mettre après upload)* |
| Pricing Monte Carlo | *(lien à mettre)* |
| Risk & VaR | *(lien à mettre)* |

> ⚠️ **Remplacer “Asset-Management-Toolbox” par le nom réel de ton repo**.  
> ⚠️ **Les liens ne fonctionneront qu'une fois les fichiers uploadés dans GitHub.**

---

## 🛠️ Installation

```bash
git clone https://github.com/hrazafimaro/Asset-Management-Toolbox.git
cd Asset-Management-Toolbox
pip install -r requirements.txt

Requirements
numpy
pandas
matplotlib
seaborn
scipy
yfinance
plotly
jupyter

📌 Utilisation du module Python (src/)
Exemple d’import :
from src.portfolio import efficient_frontier, optimize_sharpe

🎯 Objectif du projet

Ce projet a été développé pour :

montrer des compétences quantitatives (Asset Management, Finance, ML)

démontrer des capacités de structuration de projet GitHub

produire un portfolio professionnel facilement présentable en entretien

servir de base pour des projets plus avancés :
Robo-advisor, allocation dynamique, ML appliqué aux marchés, etc.

👤 Auteur

Aubin Razafimaro
Projet GitHub orienté Asset Management & Data Science.

