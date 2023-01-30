import pandas as pd


def is_dom(sol:pd.Series, solutions: pd.DataFrame):
    """
    Check if a given solution is dominated in a given dataframe

    Args:
        sol: pd.Series a specific line of solution dataframe
        solutions: datafram with all possible solutions

    Returns:
        bool That says if sol is dominated or not
    """

    for obj in ["benef","max_duration", "max_project_per_employee"]:
        obj_poss = ["benef","max_duration", "max_project_per_employee"]
        obj_poss.remove(obj)
        possible_dom = solutions[(solutions[obj_poss[0]]<=sol[obj_poss[0]]) & 
                                     (solutions[obj_poss[1]]<=sol[obj_poss[1]])]
        if sol[obj] != possible_dom[obj].min():
            return True
    return False
            
def keep_non_dom_sol(solutions:pd.DataFrame):
    """
    Select only non dominated solutions in a given dataframe

    Args:
        solutions: datafram with all possible solutions

    Returns:
        pd.DataFrame with only non dominated solutions
    """
    solutions[["benef","max_duration", "max_project_per_employee"]] = solutions[["benef","max_duration", "max_project_per_employee"]].astype(int)
    index_sol_non_dom = []
    for i in solutions.index:
        dominated = is_dom(solutions.loc[i], solutions)
        if not(dominated):
            index_sol_non_dom.append(i)
            print('Solution', i, ':',list(solutions.loc[i, ["benef","max_duration", "max_project_per_employee"]]))


    print('Gurobi found', len(index_sol_non_dom), 'non dominated solutions')
    return(solutions.loc[index_sol_non_dom])