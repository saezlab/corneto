def check_gurobi(verbose=True, solver_verbose=False):
    import contextlib
    import os
    import sys

    try:
        import gurobipy as gp
    except ImportError:
        raise ImportError("Gurobipy is not installed. Please install Gurobi and its Python bindings.")
    if verbose:
        print("Gurobipy successfully imported.")

    # Suppress stdout if solver_verbose is False
    stdout_target = sys.stdout if solver_verbose else open(os.devnull, "w")
    with contextlib.redirect_stdout(stdout_target):
        try:
            env = gp.Env(empty=True)
            env.start()
        except gp.GurobiError as e:
            raise RuntimeError(f"Error starting Gurobi environment: {e}")
        finally:
            if not solver_verbose:
                stdout_target.close()

    if verbose:
        print("Gurobi environment started successfully.")

    try:
        model = gp.Model(env=env)
        n = 3000
        x = model.addVars(n, vtype=gp.GRB.BINARY, name="x")
        model.setObjective(gp.quicksum(x[i] for i in range(n)), gp.GRB.MAXIMIZE)
        model.setParam("OutputFlag", 1 if solver_verbose else 0)

        if verbose:
            print("Starting optimization of the test model...")
        model.optimize()

        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi failed to solve the test problem optimally. Status: {model.Status}")
        if verbose:
            print("Test optimization was successful.")
    except gp.GurobiError as e:
        raise RuntimeError(f"Gurobi encountered an error during optimization: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during optimization: {e}")
    finally:
        env.dispose()
        if verbose:
            print("Gurobi environment disposed.")
    if verbose:
        print("Gurobi is correctly installed and working.")
    return True
