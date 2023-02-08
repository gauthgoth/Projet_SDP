# Projet_SDP

# Sur le projet

Ce repo git est fait dans le cadre du cours de SDP de Centrale Supélec. Il consiste à réaliser un planning pour allouer des employés à des projets en focntion de différentes contraintes. Notre but est d'optimiser le bénéfice du projet, le durée maximum d'un projet et le nombre maximum de projet par employé.

Notre rapport de projet se situe dans le notebook suivant : 

Nous avons utilisé le module gurobi pour cette optimisation. Le projet consiste en trois grande parties qui sont détaillées maintenant.

## 1. Définition des contraintes

Dans le fichier model.py, on définit le modèle gurobi ainsi que les contraintes.

## 2. Optimisation multi objectif

Dans le fichier main.py, on se sert du modèle créé auparavant pour chercher l'ensemble des solutions non dominées (en cherchant les points nadir puis en optimisant pour le reste des combinaisons et enfin en triant pour ne garder que les solutions non dominées avec les fonctions de non_domination_research.py). Nous sauvegardons les résultats au format json avec les principaux paramètres dans un csv. Le choix de l'instance se fait en changeant data_name.

Dans le notebook Viz_solution.ipynb, on peut visualiser les solutions sauvegardés auparavant dans le main. On choisit l'instance avec data_name et les fonctions de visualisations sont stockées dans utils.py .

## 3. Modèle de préférences

Dans le fichier preference.py, on monte un programme linéaire modélisant les préférences d'un décideur. Plus la classe définit est petite, plus la solution est préférée (la classe 1 est la meilleure). On a donné des exemples de préférences dans ce fichier.

