# Modules de base
import numpy as np
from tqdm import tqdm
import pandas as pd

# Module relatif à Gurobi
from gurobipy import *
from gurobipy import GRB, Model


class GurobiModel:
    """
    Class that facilitate the creation of a gurobi model, you can use it to define constraints, 
    find the nadir points and do a pre selection of potential non dominated points

    Args:
        data: dict containing the necessary information to instantiate the model
        eps: float that we use to do strict inequalities. Default = 10**(-3)

    Returns:
        None
    """
    def __init__(self, data: dict, eps:float =10**(-3)):
        self.data = data
        
        # usefull features from data
        self.n_staff = len(data["staff"])
        self.n_job = len(data["jobs"])
        self.n_days = data["horizon"]
        self.n_qual = len(data["qualifications"])

        self.eps = eps

        # model initialization
        self.m = Model("Creation de planning")

        # lists needed to save the models
        self.list_model =[]
        self.list_benef = []
        self.list_max_duration = []
        self.list_max_project_per_employee = []
        self.tuple_tested = []

        # objectives variable of the model
        self.gain = None
        self.max_duration = None
        self.max_project_per_employee = None

        # nadir points (selected after running self.find_nadir())
        self.nadir_dur = None
        self.nadir_proj = None

    def constraint_initialization(self):
        """
        Define general constraints of self.model.

        Args:
            self

        Returns:
            None
        """
        # creation de la variable de decision moelisant le probleme
        v = self.m.addMVar(shape=(self.n_staff, self.n_job, self.n_days, self.n_qual), vtype=GRB.BINARY, name="v")

        # qualification personnnel
        for i in range(self.n_staff):
            for q in range(self.n_qual):
                if not(self.data["qualifications"][q] in self.data["staff"][i]["qualifications"]):
                    self.m.addConstr(v[i,:,:,q].sum() == 0)


        # Contrainte d'unicité
        for i in range(self.n_staff):
            for t in range(self.n_days):
                self.m.addConstr(v[i,:,t,:].sum() <= 1)
                

        # Contrainte de congés
        for i in range(self.n_staff):
            for t in range(self.n_days):
                if t+1 in self.data["staff"][i]["vacations"]:
                    self.m.addConstr(v[i,:,t,:].sum() == 0)

        # Contrainte de couverture
        for p in range(self.n_job):
            for q in range(self.n_qual):
                qual = self.data["qualifications"][q]
                if qual in self.data["jobs"][p]["working_days_per_qualification"].keys():
                    self.m.addConstr(v[:,p,:,q].sum() <= self.data["jobs"][p]["working_days_per_qualification"][qual])
                    
                
                else:
                    self.m.addConstr(v[:,p,:,q].sum() <= 0)

        # def job_real
        job_real = self.m.addMVar(shape=(self.n_job), vtype=GRB.BINARY, name="job_real")
        for p in range(self.n_job):
            n_qual_need = sum(self.data["jobs"][p]["working_days_per_qualification"].values())
            self.m.addConstr(v[:,p,:,:].sum() >= n_qual_need  - (n_qual_need+ self.eps)*(1-job_real[p]))
            self.m.addConstr(v[:,p,:,:].sum() <= n_qual_need + (n_qual_need+ self.eps)*(job_real[p]))

        # def end_date
        end_date = self.m.addMVar(shape=(self.n_job), vtype=GRB.INTEGER, lb=0, ub=self.n_days, name="end_date")
        b_end_date = self.m.addMVar(shape=(self.n_job, self.n_days), vtype=GRB.BINARY, name="b_end_date")
        for p in range(self.n_job):
            n_qual_need = sum(self.data["jobs"][p]["working_days_per_qualification"].values())
            self.m.addConstr(end_date[p] <= self.n_days)
            for t in range(self.n_days):
                self.m.addConstr(v[:,p,:t+1,:].sum()>= n_qual_need -self.eps - (self.n_days + self.eps) * (1 - b_end_date[p,t]))
                self.m.addConstr(v[:,p,:t+1,:].sum()<= n_qual_need - self.eps + (self.n_days + self.eps) * b_end_date[p,t])
                
                self.m.addConstr(end_date[p] <= t * b_end_date[p,t] + self.n_days * (1 - b_end_date[p,t]))
                self.m.addConstr(end_date[p] >= (t + self.eps) * (1 - b_end_date[p,t]))

                
        # def start_date
        start_date = self.m.addMVar(shape=(self.n_job), vtype=GRB.INTEGER, lb=0, ub=self.n_days, name="start_date")
        b_start_date = self.m.addMVar(shape=(self.n_job, self.n_days), vtype=GRB.BINARY, name="b_start_date")
        for p in range(self.n_job):
            n_qual_need = sum(self.data["jobs"][p]["working_days_per_qualification"].values())
            self.m.addConstr(start_date[p] <= self.n_days)
            for t in range(self.n_days):
                self.m.addConstr(v[:,p,:t+1,:].sum()>= self.eps - (self.n_days + self.eps) * (1 - b_start_date[p,t]))
                self.m.addConstr(v[:,p,:t+1,:].sum()<= self.eps + (self.n_days + self.eps) * b_start_date[p,t])
                
                self.m.addConstr(start_date[p] <= t * b_start_date[p,t] + self.n_days * (1 - b_start_date[p,t]))
                self.m.addConstr(start_date[p] >= (t + self.eps) * (1 - b_start_date[p,t]))

        # def number of project per employee
        project_per_employee = self.m.addMVar(shape=(self.n_staff), vtype=GRB.INTEGER, lb=0, ub=self.n_job, name="project_per_employee")
        b_project_per_employee = self.m.addMVar(shape=(self.n_staff, self.n_job), vtype=GRB.BINARY, name="b_project_per_employee")
        for i in range(self.n_staff):
            self.m.addConstr(project_per_employee[i] <= self.n_job)
            for p in range(self.n_job):
                self.m.addConstr(v[i,p,:,:].sum()>= self.eps - (self.n_days + self.eps) * (1 - b_project_per_employee[i,p]))
                self.m.addConstr(v[i,p,:,:].sum()<= self.eps + (self.n_days + self.eps) * b_project_per_employee[i,p])
                
            self.m.addConstr(project_per_employee[i] == b_project_per_employee[i,:].sum())

        self.max_project_per_employee = self.m.addMVar(shape=(1), vtype=GRB.INTEGER, name="max_project_per_employee")
        self.m.addGenConstrMax(self.max_project_per_employee, list(project_per_employee))

        # def has_penality
        has_penality = self.m.addMVar(shape=(self.n_job), vtype=GRB.BINARY, name="has_penality")
        penality = self.m.addMVar(shape=(self.n_job), vtype=GRB.INTEGER, lb=0, ub=self.n_days*self.data["jobs"][0]["daily_penalty"], name="penality")
        for p in range(self.n_job):
            due_date = self.data["jobs"][p]["due_date"] -1
            self.m.addConstr(end_date[p] >= due_date + self.eps - (self.n_days + self.eps) * (1 - has_penality[p]))
            self.m.addConstr(end_date[p] <= due_date + (self.n_days + self.eps) * (has_penality[p]))
            
            self.m.addConstr(penality[p] == has_penality[p] * self.data["jobs"][p]["daily_penalty"] * (end_date[p] - due_date))


        # def duration
        duration = self.m.addMVar(shape=(self.n_job), vtype=GRB.INTEGER, name="duration")
        self.m.addConstr(duration == end_date - start_date)

        self.max_duration = self.m.addMVar(shape=(1), vtype=GRB.INTEGER, name="max_duration")
        self.m.addGenConstrMax(self.max_duration, list(duration))

        gain = np.zeros(self.n_job)
        for p in range(self.n_job):
            gain[p] = self.data["jobs"][p]["gain"]

        # def CA_per_project
        CA_per_project = self.m.addMVar(shape=(self.n_job), vtype=GRB.INTEGER, name="CA_per_project")
        self.m.addConstr(CA_per_project == job_real * (gain - penality))

        self.gain = self.m.addMVar(shape=(1), vtype=GRB.INTEGER, name="gain")
        self.m.addConstr(CA_per_project.sum() == self.gain)
        self.m.ModelSense = GRB.MINIMIZE
        
        self.m.update()
    

    def find_all_sol(self):
        """
        Pre selection of all possible non dominated points.
        You should run self.constraint_initialization() and self.find_nadir() before this function

        Args:
            self

        Returns:
            pd.DataFrame: A dataframe cointaining the values of objectives and each possible model
        """
        # iterate over all possible value in nadir set
        for max_dur in tqdm(range(self.nadir_dur+1), desc="iterate throw days"):
            for n_proj in tqdm(range(self.nadir_proj+1), desc="iterate throw jobs"):
                
                # Check if the tuple of constraint has already been optimized
                if not((max_dur, n_proj) in self.tuple_tested):
                    # copy model and set constraints
                    m_it = self.m.copy()
                    m_it.addConstr(self.max_duration == max_dur)
                    m_it.addConstr(self.max_project_per_employee == n_proj)

                    # maj
                    m_it.update()
                    m_it.optimize()
                    
                    # if solution exists, store it in a list
                    if m_it.SolCount >0 :
                        self.list_model.append(m_it)
                        self.list_benef.append(m_it.ObjVal)
                        self.list_max_duration.append(m_it.getVarByName("max_duration[0]").X)
                        self.list_max_project_per_employee.append(m_it.getVarByName("max_project_per_employee[0]").X)
        
        # create dataframe with solutions and model and drop duplicates
        df_solution = pd.DataFrame({"benef":self.list_benef,
                                "max_duration": self.list_max_duration,
                                "max_project_per_employee":self.list_max_project_per_employee,
                                "model":self.list_model})\
                                    .drop_duplicates(["benef","max_duration", "max_project_per_employee"], keep="first")
        return df_solution

    def find_nadir(self):
        """
        Find nadir points for max_duration and max_project_per_employee and print them and assign them to the object.
        You should run self.constraint_initialization() before this function

        Args:
            self

        Returns:
            None
        """
        

        # select nadir point for max_project_per_employee
        self.nadir_proj= self.epsilone_constraint_2_obj("gain", "max_duration", "max_project_per_employee")
        print("nadir projet: ", self.nadir_proj)
        
        # select nadir point for max_duration
        self.nadir_dur = self.epsilone_constraint_2_obj( "gain", "max_project_per_employee", "max_duration")
        print("nadir duration: ", self.nadir_dur)

    def epsilone_constraint_2_obj(self, obj_1:str, obj_2:str, obj_3:str):
        """
        Run epsilone constraints in order to find nadir point of obj_3

        Args:
            self
            obj_1: string initial objective of epsilone constraint, it should be "gain"
            obj_2: string second objective of epsilone constraint
            obj_3: string objective that we want to find the nadir 

        Returns:
            nadir: int containing the nadir point of obj 3
        """
        # initialize nadir to min points -1
        nadir = -1

        # create a copy of the model
        m_epsilone = self.m.copy()

        # retrieve the objectives as variables
        obj_1_var = m_epsilone.getVarByName(obj_1 + "[0]")
        obj_2_var = m_epsilone.getVarByName(obj_2+ "[0]")
        obj_3_var = m_epsilone.getVarByName(obj_3 + "[0]")

        # set the objectives with the right priorities (we change the size for gain)
        m_epsilone.setObjectiveN(-obj_1_var, 1, priority=2)
        m_epsilone.setObjectiveN(obj_2_var, 2, priority=1)
        m_epsilone.setObjectiveN(obj_3_var, 3, priority=0)
        model_has_sol = True
        m_epsilone.update()
        
        # epsilone constraint for obj1 and obj_2
        while tqdm(model_has_sol, desc="finding nadir for " + obj_3):
            m_epsilone_it = m_epsilone.copy()
            m_epsilone_it.update()
            m_epsilone_it.optimize()
            
            # check if solution exists
            if m_epsilone_it.SolCount >0:
                # store solution in a list
                self.list_model.append(m_epsilone_it)
                self.list_benef.append(-m_epsilone_it.getVarByName("gain[0]").X)
                self.list_max_duration.append(m_epsilone_it.getVarByName("max_duration[0]").X)
                self.list_max_project_per_employee.append(m_epsilone_it.getVarByName("max_project_per_employee[0]").X)
                self.tuple_tested.append((m_epsilone_it.getVarByName("max_duration[0]").X, 
                m_epsilone_it.getVarByName("max_project_per_employee[0]").X))

                # select the max value of obj_3 as nadir throw the loops
                nadir = max([nadir, m_epsilone_it.getVarByName(obj_3 + "[0]").X])

                # set the epsilone constraint
                m_epsilone.addConstr(obj_2_var <= m_epsilone_it.getVarByName(obj_2 + "[0]").X - self.eps)
                m_epsilone.update()
            else: model_has_sol=False # if no solution, stop the loop
        return int(nadir)

