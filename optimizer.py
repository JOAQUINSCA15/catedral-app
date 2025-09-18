
import pulp
import time

def build_and_solve_shift_model(W, D, T, S, MinHw, MaxHw, Demand, Max_Deviation=2.5, time_limit=120):
    model = pulp.LpProblem("Shift_Scheduling", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("x", (W,D,T), cat="Binary")
    y = pulp.LpVariable.dicts("y", (W,D), cat="Binary")
    z = pulp.LpVariable.dicts("z", (W,range(1,7)), cat="Binary")
    b = pulp.LpVariable.dicts("b", (W,D,S), cat="Binary")
    under = pulp.LpVariable.dicts("under", (D,T), lowBound=0)
    over = pulp.LpVariable.dicts("over", (D,T), lowBound=0)

    model += pulp.lpSum([under[d][t] + over[d][t] for d in D for t in T])

    for w in W:
        for d in D:
            model += pulp.lpSum([x[w][d][t] for t in T]) >= 4*y[w][d]
            model += pulp.lpSum([x[w][d][t] for t in T]) <= 8*y[w][d]

    for w in W:
        model += pulp.lpSum([x[w][d][t] for d in D for t in T]) >= MinHw[w]
        model += pulp.lpSum([x[w][d][t] for d in D for t in T]) <= MaxHw[w]

    for w in W:
        model += pulp.lpSum([z[w][d] for d in range(1,7)]) == 1
        for d in range(1,7):
            model += z[w][d] <= 1 - y[w][d]
            model += z[w][d] <= 1 - y[w][d+1]

    for w in W:
        for d in range(1,7):
            for t in range(13,16):
                model += x[w][d][t] + x[w][d+1][t-12] <= 1

    for w in W:
        model += pulp.lpSum([x[w][d][15] for d in D]) <= 2

    for d in D:
        for t in T:
            model += pulp.lpSum([x[w][d][t] for w in W]) + under[d][t] - over[d][t] == Demand[d][t-1]
            model += under[d][t] + over[d][t] <= Max_Deviation
            model += pulp.lpSum([x[w][d][t] for w in W]) >= 1

    for w in W:
        for d in D:
            model += pulp.lpSum([b[w][d][s,e] for (s,e) in S]) <= 1
            for t in T:
                model += x[w][d][t] == pulp.lpSum([b[w][d][s,e] for (s,e) in S if s<=t<=e])
            model += y[w][d] == pulp.lpSum([b[w][d][s,e] for (s,e) in S])

    for w in W:
        if MinHw[w] == 15:
            for d in D:
                if d not in [5,6,7]:
                    model += y[w][d] == 0

    start_time = time.time()
    result_status = model.solve(pulp.PULP_CBC_CMD(msg=True, timeLimit=time_limit))
    end_time = time.time()

    schedule = []
    for w in W:
        for d in D:
            for t in T:
                if pulp.value(x[w][d][t]) > 0.5:
                    schedule.append((w, d, t))

    return {
        "status": pulp.LpStatus[model.status],
        "objective": pulp.value(model.objective),
        "elapsed_time": end_time - start_time,
        "schedule": schedule
    }
