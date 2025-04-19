from ortools.linear_solver import pywraplp


def create_data_model():
    """Create the data for the example."""
    data = {}
    weights = [0.48, 0.30, 0.19, 0.36, 0.36, 0.27, 0.42, 0.42, 0.36, 0.24, 0.30]
    data["resource_need"] = weights
    data["vms"] = list(range(len(weights)))
    # 10_000 machines -> 2.5mins
    data["machines"] = list(range(100_000))
    data["bin_capacity"] = 1
    return data



def main():
    data = create_data_model()

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP")

    if not solver:
        return

    # Variables
    # x[i, j] = 1 if item i is packed in bin j.
    x = {}
    for i in data["vms"]:
        for j in data["machines"]:
            x[(i, j)] = solver.IntVar(0, 1, "x_%i_%i" % (i, j))

    # y[j] = 1 if bin j is used.
    y = {}
    for j in data["machines"]:
        y[j] = solver.IntVar(0, 1, "y[%i]" % j)

    # Constraints
    # Each item must be in exactly one bin.
    for i in data["vms"]:
        solver.Add(sum(x[i, j] for j in data["machines"]) == 1)

    # The amount packed in each bin cannot exceed its capacity.
    for j in data["machines"]:
        solver.Add(
            sum(x[(i, j)] * data["resource_need"][i] for i in data["vms"])
            <= y[j] * data["bin_capacity"]
        )

    # Objective: minimize the number of machines used.
    solver.Minimize(solver.Sum([y[j] for j in data["machines"]]))

    print(f"Solving with {solver.SolverVersion()}")
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        num_bins = 0
        for j in data["machines"]:
            if y[j].solution_value() == 1:
                bin_items = []
                bin_weight = 0
                for i in data["vms"]:
                    if x[i, j].solution_value() > 0:
                        bin_items.append(i)
                        bin_weight += data["resource_need"][i]
                if bin_items:
                    num_bins += 1
                    print("Bin number", j)
                    print("  VMs packed:", bin_items)
                    print("  Total resources used:", bin_weight)
                    print()
        print()
        print("Number of machines used:", num_bins)
        print("Time = ", solver.WallTime(), " milliseconds")
    else:
        print("The problem does not have an optimal solution.")


if __name__ == "__main__":
    main()

