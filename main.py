from model import GurobiModel
from non_domination_research import keep_non_dom_sol
from utils import save_models_mps_sol, save_models_json
from gurobipy import GRB

import json
import os

data_name = "toy_instance"

# Open the json
with open(data_name + '.json') as json_file:
    data = json.load(json_file)

# initiate the model with the given data
model = GurobiModel(data)

# initialize constraint and zero flag the model
model.constraint_initialization()
model.m.params.outputflag = 0
model.m.update()

# find nadir points for max_duration and max_project_per_employee
model.find_nadir()

# set the gain as objective
model.m.setObjective(-model.gain, GRB.MINIMIZE)
model.m.update()

# find all solutions within nadir space
df_solution = model.find_all_sol()

# keep only dominated 
df = keep_non_dom_sol(df_solution)

# save the model
#save_models_mps_sol(df, data_name)
save_models_json(df,data_name)