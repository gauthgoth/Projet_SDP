import numpy as np
from gurobipy import *
from gurobipy import Model, GRB
import pandas as pd
import os

data_name = "medium_instance"
df = pd.read_csv(os.path.join("solution", data_name, "df_solution_json.csv"), index_col="index")

# preference du decideur (doit regrouper les classes entres elles )
if data_name == "toy_instance":

    reference_decideur = -np.array([ [-65, 1, 3, 1],
                                    [-65, 2, 2, 1],
                                    [-55, 1, 2, 2],
                                    [0, 0, 0, 3],
                                    [0, 0, 0, 3]])
elif data_name == "medium_instance":
    reference_decideur = -np.array([[-413, 7, 6, 1],
                                    [-130, 5, 1, 2],
                                    [0, 0, 0, 3],])
else:
    reference_decideur = -np.array([[]
                                    ])
new_planning = -np.array(df.iloc[:, : 3])


####### Param du probleme
Li=3
n_criter=3
eps=10**(-3)

########Compute le score 
def x(i:int, k:int):
    return reference_decideur[:,i].min() + (k/Li) * (reference_decideur[:,i].max() - reference_decideur[:,i].min())

def k_x(i:int, j:int, reference:np.array):
    k = 0
    while (x(i,k) <= reference[j,i]) and (k!=Li) :
        k+=1
    return k-1

def s_i(i:int, j:int, reference:np.array, eval:bool=False):
    k = k_x(i, j, reference)
    coef = (reference[j,i] - x(i,k)) / (x(i,k+1) - x(i,k))
    if eval:
        return v.X[k,i] + coef * (v.X[k+1,i] - v.X[k,i])
    return v[k,i] + coef * (v[k+1,i] - v[k,i])

def score(j:int, reference:np.array, eval:bool=False):
    sum_ = 0
    for i in range(n_criter):
        sum_+=s_i(i,j, reference, eval)
    return sum_

def classe(j:int, reference:np.array):
    return reference[j][n_criter]


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

# define the constraints that rank solution in the point of view of the decision maker
for j in range(len(reference_decideur)-1):
    c=classe(j, reference_decideur)
    c_=classe(j+1, reference_decideur)
    if c_<c:
        m.addConstr(score(j, reference_decideur) - sig[j]["sur"] + sig[j]["sous"] >= 
                    score(j+1, reference_decideur) - sig[j+1]["sur"] + sig[j+1]["sous"]+eps)
    elif c<c_:
        m.addConstr(score(j+1, reference_decideur) - sig[j+1]["sur"] + sig[j+1]["sous"] >= 
                    score(j, reference_decideur) - sig[j]["sur"] + sig[j]["sous"]+eps)
    else:
        m.addConstr(score(j, reference_decideur) - score(j+1, reference_decideur) + 
                    sig[j]["sous"] - sig[j+1]["sous"] + sig[j+1]["sur"] - sig[j]["sur"] == 0)

# set the overall error as objective to minimize
obj = 0
for j in range(len(reference_decideur)):
    obj+=sig[j]["sur"] + sig[j]["sous"]
m.params.outputflag = 0

m.setObjective(obj, GRB.MINIMIZE)

m.update()

# Resolution du PL
m.optimize()

# print error
print(m.ObjVal)


def rank_ref(reference:np.array, with_error:bool):
    dict_val = dict()
    for i in range(len(reference)):
        if with_error:
            dict_val[i] = score(i, reference, eval=True) - sig[i]["sur"].X + sig[i]["sous"].X
        else:
            dict_val[i] = score(i, reference, eval=True)
    return dict(sorted(dict_val.items(), key=lambda item: item[1], reverse=True))


print(rank_ref(reference_decideur, with_error=True))

print(rank_ref(reference_decideur, with_error=False))

print(rank_ref(new_planning, with_error=False))

