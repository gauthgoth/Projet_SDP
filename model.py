# Modules de base
import numpy as np
from tqdm import tqdm
import pandas as pd

# Module relatif à Gurobi
from gurobipy import *
from gurobipy import GRB, Model


class GurobiModel:
    def __init__(self, data, eps=10**(-3)):
        self.data = data

        self.n_staff = len(data["staff"])
        self.n_job = len(data["jobs"])
        self.n_days = data["horizon"]
        self.n_qual = len(data["qualifications"])

        self.eps = eps

        self.m = Model("Creation de planning")


        self.list_model =[]
        self.list_benef = []
        self.list_max_duration = []
        self.list_max_project_per_employee = []
        self.tuple_tested = []

        self.gain = None
        self.max_duration = None
        self.max_project_per_employee = None

    def constraint_initialization(self):
        # Création d'un vecteur de 3 variables continues
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

        # CA_per_project
        CA_per_project = self.m.addMVar(shape=(self.n_job), vtype=GRB.INTEGER, name="CA_per_project")
        self.m.addConstr(CA_per_project == job_real * (gain - penality))

        self.gain = self.m.addMVar(shape=(1), vtype=GRB.INTEGER, name="gain")
        self.m.addConstr(CA_per_project.sum() == self.gain)
        self.m.ModelSense = GRB.MINIMIZE
        
        self.m.update()
    

    def find_all_sol(self):
        for max_dur in tqdm(range(self.nadir_dur+1), desc="iterate throw days"):
            for n_proj in tqdm(range(self.nadir_proj+1), desc="iterate throw jobs"):
                if not((max_dur, n_proj) in self.tuple_tested):
                    m_it = self.m.copy()
                    m_it.addConstr(self.max_duration == max_dur)
                    m_it.addConstr(self.max_project_per_employee == n_proj)
                
                    # maj
                    m_it.update()
                    # Affichage en mode texte du PL
                    m_it.optimize()

                    if m_it.SolCount >0 :
                        self.list_model.append(m_it)
                        self.list_benef.append(-m_it.ObjVal)
                        self.list_max_duration.append(m_it.getVarByName("max_duration[0]").X)
                        self.list_max_project_per_employee.append(m_it.getVarByName("max_project_per_employee[0]").X)
        df_solution = pd.DataFrame({"benef":self.list_benef,
                                "max_duration": self.list_max_duration,
                                "max_project_per_employee":self.list_max_project_per_employee,
                                "model":self.list_model})\
                                    .drop_duplicates(["benef","max_duration", "max_project_per_employee"], keep="first")
        return df_solution

    def find_nadir(self):
        m_nad_dur_ca = self.m.copy()
        gain = m_nad_dur_ca.getVarByName("gain[0]")
        max_duration = m_nad_dur_ca.getVarByName("max_duration[0]")
        max_project_per_employee = m_nad_dur_ca.getVarByName("max_project_per_employee[0]")
        m_nad_dur_ca.setObjectiveN(-gain, 0, priority=1)
        m_nad_dur_ca.setObjectiveN(max_duration, 1, priority=2)
        m_nad_dur_ca.setObjectiveN(max_project_per_employee, 2, priority=0)
        m_nad_dur_ca.update()
        m_nad_dur_ca.optimize()
        m_nad_dur_ca.params.ObjNumber = 0
        self.list_model.append(m_nad_dur_ca)
        self.list_benef.append(-m_nad_dur_ca.ObjVal)
        self.list_max_duration.append(m_nad_dur_ca.getVarByName("max_duration[0]").X)
        self.list_max_project_per_employee.append(m_nad_dur_ca.getVarByName("max_project_per_employee[0]").X)
        self.tuple_tested.append((m_nad_dur_ca.getVarByName("max_duration[0]").X, 
        m_nad_dur_ca.getVarByName("max_project_per_employee[0]").X))
        m_nad_dur_ca.params.ObjNumber = 2

        m_nad_ca_dur = self.m.copy()
        gain = m_nad_ca_dur.getVarByName("gain[0]")
        max_duration = m_nad_ca_dur.getVarByName("max_duration[0]")
        max_project_per_employee = m_nad_ca_dur.getVarByName("max_project_per_employee[0]")
        m_nad_ca_dur.setObjectiveN(-gain, 0, priority=2)
        m_nad_ca_dur.setObjectiveN(max_duration, 1, priority=1)
        m_nad_ca_dur.setObjectiveN(max_project_per_employee, 2, priority=0)
        m_nad_ca_dur.update()
        m_nad_ca_dur.optimize()
        m_nad_ca_dur.params.ObjNumber = 0
        self.list_model.append(m_nad_ca_dur)
        self.list_benef.append(-m_nad_ca_dur.ObjVal)
        self.list_max_duration.append(m_nad_ca_dur.getVarByName("max_duration[0]").X)
        self.list_max_project_per_employee.append(m_nad_ca_dur.getVarByName("max_project_per_employee[0]").X)
        self.tuple_tested.append((m_nad_ca_dur.getVarByName("max_duration[0]").X, 
        m_nad_ca_dur.getVarByName("max_project_per_employee[0]").X))
        m_nad_ca_dur.params.ObjNumber = 2

        self.nadir_proj= int(max([m_nad_dur_ca.ObjNVal, m_nad_ca_dur.ObjNVal]))
        print("nadir projet: ", self.nadir_proj)

        m_nad_proj_ca = self.m.copy()
        gain = m_nad_proj_ca.getVarByName("gain[0]")
        max_duration = m_nad_proj_ca.getVarByName("max_duration[0]")
        max_project_per_employee = m_nad_proj_ca.getVarByName("max_project_per_employee[0]")
        m_nad_proj_ca.setObjectiveN(-gain, 0, priority=1)
        m_nad_proj_ca.setObjectiveN(max_duration, 1, priority=0)
        m_nad_proj_ca.setObjectiveN(max_project_per_employee, 2, priority=2)
        m_nad_proj_ca.update()
        m_nad_proj_ca.optimize()
        m_nad_proj_ca.params.ObjNumber = 0
        self.list_model.append(m_nad_proj_ca)
        self.list_benef.append(-m_nad_proj_ca.ObjVal)
        self.list_max_duration.append(m_nad_proj_ca.getVarByName("max_duration[0]").X)
        self.list_max_project_per_employee.append(m_nad_proj_ca.getVarByName("max_project_per_employee[0]").X)
        self.tuple_tested.append((m_nad_proj_ca.getVarByName("max_duration[0]").X, 
        m_nad_proj_ca.getVarByName("max_project_per_employee[0]").X))
        m_nad_proj_ca.params.ObjNumber = 1

        

        m_nad_ca_proj = self.m.copy()
        gain = m_nad_ca_proj.getVarByName("gain[0]")
        max_duration = m_nad_ca_proj.getVarByName("max_duration[0]")
        max_project_per_employee = m_nad_ca_proj.getVarByName("max_project_per_employee[0]")
        m_nad_ca_proj.setObjectiveN(-gain, 0, priority=2)
        m_nad_ca_proj.setObjectiveN(max_duration, 1, priority=0)
        m_nad_ca_proj.setObjectiveN(max_project_per_employee, 2, priority=1)
        m_nad_ca_proj.update()
        m_nad_ca_proj.optimize()
        m_nad_ca_proj.params.ObjNumber = 0
        self.list_model.append(m_nad_ca_proj)
        self.list_benef.append(-m_nad_ca_proj.ObjVal)
        self.list_max_duration.append(m_nad_ca_proj.getVarByName("max_duration[0]").X)
        self.list_max_project_per_employee.append(m_nad_ca_proj.getVarByName("max_project_per_employee[0]").X)
        self.tuple_tested.append((m_nad_ca_proj.getVarByName("max_duration[0]").X, 
        m_nad_ca_proj.getVarByName("max_project_per_employee[0]").X))
        m_nad_ca_proj.params.ObjNumber = 1


        self.nadir_dur= int(max([m_nad_proj_ca.ObjNVal, m_nad_ca_proj.ObjNVal]))
        print("nadir duration: ", self.nadir_dur)

        