from model import GurobiModel
from non_domination_research import keep_non_dom_sol
from utils import save_models_mps_sol, save_models_json
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

#save_models_mps_sol(df, data_name)
save_models_json(df,data_name)