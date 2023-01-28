import numpy as np
from gurobipy import *
from gurobipy import Model, GRB
import pandas as pd
import os

data_name = "toy_instance"
df = pd.read_csv(os.path.join("solution", data_name, "df_solution_json.csv"), index_col="index")

reference_decideur=-np.array([ [-65, 1, 3, 1],
                                [-65, 2, 2, 1],
                                [-55, 1, 2, 2],
                                [0, 0, 0, 3],
                                [0, 0, 0, 3]])
new_planning=-np.array(df.iloc[:, : 3])


####### Param du probleme
Li=3
n_criter=3
eps=10**(-3)

########Compute le score 
def x(i,k):
    return reference_decideur[:,i].min() + (k/Li) * (reference_decideur[:,i].max() - reference_decideur[:,i].min())

def k_x(i,j):
    k = 0
    while (x(i,k) <= reference_decideur[j,i]) and (k!=Li) :
        k+=1
    return k-1

def s_i(i,j):
    k = k_x(i,j)
    coef = (reference_decideur[j,i] - x(i,k)) / (x(i,k+1) - x(i,k))
    return v[k,i] + coef * (v[k+1,i] - v[k,i])

def score(j):
    sum_ = 0
    for i in range(n_criter):
        sum_+=s_i(i,j)
    return sum_

def classe(j):
    return reference_decideur[j][n_criter]


#####Instanciation du modele
m=Model("Systeme de preference")


##### Variables de decisions
v = m.addMVar(shape=(Li+1,n_criter))

#Erreurs:
sig = dict()
for j in range(len(reference_decideur)):
    sig[j] = {"sur": m.addVar(), "sous": m.addVar()}
    
m.update()

# Contraintes sur les ui(gi)
# Les criteres sont complementaires: 
m.addConstr(v[Li].sum() == 1)

# On commence en 0:
m.addConstr(v[0] == np.zeros(n_criter))    

# Les scores sont croissants
for i in range(Li):
    m.addConstr(v[i] <= v[i+1])
    
for j in range(len(reference_decideur)-1):
    c=classe(j)
    c_=classe(j+1)
    if c_<c:
        m.addConstr(score(j) - sig[j]["sur"] + sig[j]["sous"] >= score(j+1) - sig[j+1]["sur"] + sig[j+1]["sous"]+eps)
    elif c<c_:
        m.addConstr(score(j+1) - sig[j+1]["sur"] + sig[j+1]["sous"] >= score(j) - sig[j]["sur"] + sig[j]["sous"]+eps)
    else:
        m.addConstr(score(j)-score(j+1)+sig[j]["sous"]-sig[j+1]["sous"]+sig[j+1]["sur"]-sig[j]["sur"] == 0)

obj = 0
for j in range(len(reference_decideur)):
    obj+=sig[j]["sur"] + sig[j]["sous"]
# Parametrage (mode mute)
m.params.outputflag = 0

m.setObjective(obj, GRB.MINIMIZE)

m.update()

# Resolution du PL
m.optimize()

print(m.ObjVal)

def k_x_eval(i,j, list_alternatives):
    k = 0
    while (x(i,k) <= list_alternatives[j,i]) and (k!=Li) :
        k+=1
    return k-1

def s_i_eval(i,j, list_alternatives):
    k = k_x_eval(i,j, list_alternatives)
    coef = (list_alternatives[j,i] - x(i,k)) / (x(i,k+1) - x(i,k))
    return v.X[k,i] + coef * (v.X[k+1,i] - v.X[k,i])

def s_eval(j, list_alternatives):
    sum_ = 0
    for i in range(n_criter):
        sum_+=s_i_eval(i,j, list_alternatives)
    return sum_

def rank_ref(list_alternatives):
    dict_val = dict()
    for i in range(len(list_alternatives)):
        dict_val[i] = s_eval(i, list_alternatives) - sig[i]["sur"].X + sig[i]["sous"].X
    return dict(sorted(dict_val.items(), key=lambda item: item[1], reverse=True))

def rank_new(list_alternatives):
    dict_val = dict()
    for i in range(len(list_alternatives)):
        dict_val[i] = s_eval(i, list_alternatives)
    
    return dict(sorted(dict_val.items(), key=lambda item: item[1], reverse=True))

print(rank_ref(reference_decideur))

print(rank_new(reference_decideur))

print(rank_new(new_planning))

print(v.X)