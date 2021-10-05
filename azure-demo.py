from demo import *
from azure.quantum import Workspace
from azure.quantum.optimization import Problem, Term, ProblemType, SimulatedAnnealing
if __name__ == '__main__':

    # Set up scenario
    num_pumps = 7
    pumps = ['P'+str(p+1) for p in range(num_pumps)]
    time = list(range(1, 25))
    power = [15, 37, 33, 33, 22, 33, 22]
    costs = [169]*7 + [283]*6 + [169]*3 + [336]*5 + [169]*3
    flow = [75, 133, 157, 176, 59, 69, 120]
    demand = [44.62, 31.27, 26.22, 27.51, 31.50, 46.18, 69.47, 100.36, 131.85, 
                148.51, 149.89, 142.21, 132.09, 129.29, 124.06, 114.68, 109.33, 
                115.76, 126.95, 131.48, 138.86, 131.91, 111.53, 70.43]
    v_init = 550
    v_min = 523.5
    v_max = 1500
    c3_gamma = 0.00052

    # Build BQM
    bqm, x = build_bqm(num_pumps, time, power, costs, flow, demand, v_init, v_min, v_max, c3_gamma)

    def from_bqm(bqm):
        terms = []

        # Create a dictionary of variable names to indices
        index_mappings = dict()

        for variable_name in bqm.variables:
            index_mappings[variable_name] = len(index_mappings)

        # Add constant value
        terms += [Term(c=bqm.offset, indices=[])]

        # Add linear terms
        for variable in bqm.linear:
            index = index_mappings[variable]
            value = bqm.linear[variable]

            terms += [Term(c=value, indices=[index])]

        # Add quadratic terms
        for (var1, var2) in bqm.quadratic:
            id1 = index_mappings[var1]
            id2 = index_mappings[var2]

            value = bqm.quadratic[(var1, var2)]

            terms += [Term(c=value, indices=[id1, id2])]

        if bqm.vartype == "SPIN":
            return Problem(name="bqm", terms=terms, problem_type=ProblemType.ising)
        else:
            return Problem(name="bqm", terms=terms, problem_type=ProblemType.pubo)

    def update_variables(config, bqm):
        # Create a dictionary of variable names to indices
        bqm_mappings = dict()

        for variable_name in bqm.variables:
            bqm_mappings[str(len(bqm_mappings))] = variable_name

        sample = dict()

        for c in config:
            sample[bqm_mappings[c]] = config[c]

        return sample         

    # Log in to Azure Quantum workspace
    workspace = Workspace (
        subscription_id = "",
        resource_group = "",
        name = "",
        location = ""
    )

    # Convert the BQM and create an Azure Quantum Problem
    problem = from_bqm(bqm)

    # Select an Azure Quantum solver
    solver = SimulatedAnnealing(workspace, timeout=100)

    # Submit the problem
    print("\nRunning Azure Quantum solver...")
    result = solver.optimize(problem)
    config = result["configuration"]
    sample = update_variables(config, bqm)

    # Process-lowest energy solution
    pump_flow_schedule, reservoir = process_sample(sample, x, pumps, time, power, flow, costs, demand, v_init)

    # Visualize result
    visualize(sample, x, v_min, v_max, v_init, num_pumps, costs, power, pump_flow_schedule, reservoir, time, demand)
