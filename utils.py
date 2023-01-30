import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from gurobipy import read

def save_models_mps_sol(df:pd.DataFrame, data_name: str):
    """
    Save model stored in column model in .mps and .store format. Add the columns of the path in df

    Args:
        df: pd.DataFrame output of keep_non_dom_sol
        data_name: string name of the data that we used to create the df

    Returns:
        df: pd.DataFrame A dataframe with the columns model_path added
    """
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
    'model_path_sol', "model_path_mps"]].to_csv(os.path.join("solution", data_name, "df_solution_mps_sol.csv"), index_label="index")

    return df

def load_models_mps_sol(df:pd.DataFrame):
    """
    Read the save models ond optimize them

    Args:
        df: pd.DataFrame output of save_models_mps_sol

    Returns:
        df: pd.DataFrame A dataframe with the optimized model in column path
    """
    list_model = []
    for index in tqdm(df.index, desc = "iterating throw indexes"):
        model = read(df.loc[index, "model_path_mps"])
        model.read(df.loc[index, "model_path_sol"])
        model.params.outputflag = 0
        model.update()
        model.optimize()
        list_model.append(model)

    df["model"] = list_model

    return df

def save_models_json(df:pd.DataFrame, data_name:str):
    """
    Save model stored in column model in .json format. Add the columns of the path in df

    Args:
        df: pd.DataFrame output of keep_non_dom_sol
        data_name: string name of the data that we used to create the df

    Returns:
        df: pd.DataFrame A dataframe with the columns model_path added
    """
    path = os.path.join("solution", data_name, 'model')
    os.makedirs(path, exist_ok=True)
    model_path_json = []
    for index in df.index:
        df.loc[index, "model"].write(os.path.join(path, str(index) + ".json"))
        model_path_json.append(os.path.join(path, str(index) + ".json"))

    df["model_path_json"] = model_path_json

    df[['benef', 'max_duration', 'max_project_per_employee', 
    'model_path_json']].to_csv(os.path.join("solution", data_name, "df_solution_json.csv"), index_label="index")

    return df

class SolutionFromDict:
    """
    Class that permits to vizualize planning from given dictionnaries

    Args:
        data: dict that gives general information of the instance
        sol: dict output of the write with a .json from a gurobi model
    """
    def __init__(self, data:dict, sol:dict):
        self.data = data

        # general info used in the viz
        self.n_staff = len(data["staff"])
        self.n_job = len(data["jobs"])
        self.n_days = data["horizon"]
        self.n_qual = len(data["qualifications"])
        self.list_staff = [staff["name"] for staff in data["staff"]]
        self.list_job = [staff["name"] for staff in data["jobs"]]
        self.list_qual = self.data["qualifications"]

        self.sol = sol

        # main decision variable
        self.v = None

        # info on objectives
        try:
            self.gain = next(item["X"] for item in self.sol["Vars"] if item["VarName"] == "gain[0]")
        except:
            self.gain = 0
        try:
            self.max_duration = next(item["X"] for item in self.sol["Vars"] if item["VarName"] == "max_duration[0]")
        except:
            self.max_duration = 0
        try:
            self.max_project_per_employee = next(item["X"] for item in self.sol["Vars"] if item["VarName"] == "max_project_per_employee[0]")
        except:
            self.max_project_per_employee = 0
            
        


    def get_v(self):
        """
        Retrieve all v values and store them in self.v

        Args:
            self
        """
        self.v = np.zeros((self.n_staff, self.n_job, self.n_days, self.n_qual))
        for i in range(self.n_staff):
            for p in range(self.n_job):
                for t in range(self.n_days):
                    for q in range(self.n_qual):
                        try :
                            self.v[i,p,t,q] = next(item["X"] for item in self.sol["Vars"] if item["VarName"] == "v[%d,%d,%d,%d]"%(i,p,t,q))
                        except:
                            pass
        return self.v
    
    def get_CA_per_project(self):
        """
        Retrieve all values of CA;project and store them in self.CA_per_project
        
        Args:
            self

        Returns:
            self.CA_per_project: np.array containing the CA of each project
        """
        self.CA_per_project = np.zeros(self.n_job)
        for p in range(self.n_job):
            try :
                self.CA_per_project[p] = next(item["X"] for item in self.sol["Vars"] if item["VarName"] == "CA_per_project[%d]"%(p))
            except:
                pass
        return self.CA_per_project
    
    def get_planning_employee(self, employee:str):
        """
        select an employee and give his planning with the jobs where he works and qualification needed
        
        Args:
            self
            employee: str employee name

        Returns:
            planning: list. Planning of the employee per day (1 list element is a day)
        """
        i = self.list_staff.index(employee)
        planning = []
        for t in range(self.n_days):
            day_t = []
            for p in range(self.n_job):
                for q in range(self.n_qual):
                    if self.v[i,p,t,q] >0.5:
                        day_t.append(self.list_job[p] + " with qualification " + self.list_qual[q])
            if len(day_t)==0:
                day_t.append("vacation")
            planning.append(day_t)
        return planning
        


    def get_planning_project(self, project:str):
        """
        select a project and give the timetable of a job with the employees sent and their qualifcations apllied
        
        Args:
            self
            project: str project name

        Returns:
            planning: list. Planning of the project per day (1 list element is a day)
        """
        p = self.list_job.index(project)
        planning = []
        for t in range(self.n_days):
            day_t = []
            for i in range(self.n_staff):
                for q in range(self.n_qual):
                    if self.v[i,p,t,q] >0.5:
                        day_t.append(self.list_staff[i] + " with qualification " + self.list_qual[q])
            if len(day_t)==0:
                day_t.append("None")
            planning.append(day_t)
        return planning
        
