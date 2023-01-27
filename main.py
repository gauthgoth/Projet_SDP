from model import GurobiModel
from non_domination_research import keep_non_dom_sol
from gurobipy import GRB

import json
import os

data_name = "toy_instance"

# Opening JSON file
with open(data_name + '.json') as json_file:
    data = json.load(json_file)

model = GurobiModel(data)

model.constraint_initialization()
model.m.params.outputflag = 0
model.m.update()

model.find_nadir()


model.m.setObjective(-model.gain, GRB.MINIMIZE)
model.m.update()

df_solution = model.find_all_sol()

indexes, df = keep_non_dom_sol(df_solution)

path = os.path.join("solution", data_name, 'model')
os.makedirs(path, exist_ok=True)
model_path_mps = []
model_path_sol = []
for index in df.index:
    df.loc[index, "model"].write(os.path.join(path, str(index) + ".mps"))
    model_path_mps.append(os.path.join(path, str(index) + ".mps"))
    
    df.loc[index, "model"].write(os.path.join(path, str(index) + ".sol"))
    model_path_sol.append(os.path.join(path, str(index) + ".sol"))

df["model_path_mps"] = model_path_mps
df["model_path_sol"] = model_path_sol

df[['benef', 'max_duration', 'max_project_per_employee', 
    'model_path_sol', "model_path_mps"]].to_csv(os.path.join("solution", data_name, "df_solution.csv"), index_label="index")