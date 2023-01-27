
def is_dom(sol, solutions):
    for obj in ["benef","max_duration", "max_project_per_employee"]:
        obj_poss = ["benef","max_duration", "max_project_per_employee"]
        obj_poss.remove(obj)
        possible_dom = solutions[(solutions[obj_poss[0]]<=sol[obj_poss[0]]) & 
                                     (solutions[obj_poss[1]]<=sol[obj_poss[1]])]
        #print(possible_dom, sol, obj_poss)
        if sol[obj] != possible_dom[obj].min():
            return True
    return False
            
def keep_non_dom_sol(solutions):
    index_sol_non_dom = []
    for i in solutions.index:
        dominated = is_dom(solutions.loc[i], solutions)
        if not(dominated):
            index_sol_non_dom.append(i)
            print('Solution', i, ':',list(solutions.loc[i, ["benef","max_duration", "max_project_per_employee"]]))


    print('Gurobi found', len(index_sol_non_dom), 'non dominated solutions')
    return(index_sol_non_dom, solutions.loc[index_sol_non_dom])