
import cplex

class Request:
    def __init__(self, a, d,S,e,b):
        self.a = a
        self.d = d
        self.S = S
        self.e = e
        self.b = b

def cplex_solve_question(N,K,P,R,Q,C,Request_list,T):

    varnames=[]
    ub=[]
    lb=[]
    obj=[]
    types=""
    for i in range(1,N+1):
        for j in range(1,T+1):
            varnames.append("x_"+str(i)+"_"+str(j))
            ub.append(1)
            lb.append(0)
            types=types+"I"
            obj.append(Request_list[i-1].b)
    for i in range(1,N+1):
        for j in range(1,K+1):
            for k in range(1,P+1):
                for l in range(1,T+1):
                    varnames.append("y_"+str(i)+"_"+str(j)+"_"+str(k)+"_"+str(l))
                    ub.append(cplex.infinity)
                    lb.append(0)
                    types = types + "I"
                    obj.append(0)

    senses = ""
    rhs = []
    rownames = []
    rows = []
    for i in range(1,N+1):
        for j in range(1,K+1):
            for k in range(1,T+1):
                row_varnames=[]
                row_factor=[]
                for l in range(max(k-Request_list[i-1].e+1,1),k+1):
                    row_varnames.append("x_"+str(i)+"_"+str(l))
                    row_factor.append(Request_list[i-1].S[j-1][k+1-l-1])
                for l in range(1,P+1):
                    row_varnames.append("y_" + str(i) + "_" + str(j)+ "_" + str(l)+ "_" + str(k))
                    row_factor.append(-1)
                rownames.append("ra_" + str(i) + "_" + str(j) + "_" + str(k))
                rows.append([row_varnames, row_factor])
                senses=senses+"E"
                rhs.append(0)

    for i in range(1,P+1):
        for j in range(1,T+1):
            for k in range(1,R+1):
                row_varnames = []
                row_factor = []
                for l in range(1,N+1):
                    for p in range(1,K+1):
                        row_varnames.append("y_"+str(l)+"_"+str(p)+"_"+str(i)+"_"+str(j))
                        row_factor.append(Q[p-1][k-1])
                rownames.append("rb_" + str(i) + "_" + str(j) + "_" + str(k))
                rows.append([row_varnames, row_factor])
                senses = senses + "L"
                rhs.append(C[i-1][k-1])

    for i in range(1,N+1):
        row_varnames = []
        row_factor = []
        for j in range(Request_list[i-1].a,Request_list[i-1].d-Request_list[i-1].e+2):
            row_varnames.append("x_" + str(i) + "_" + str(j) )
            row_factor.append(1)
        rownames.append("rc_" + str(i) )
        rows.append([row_varnames, row_factor])
        senses = senses + "L"
        rhs.append(1)

    for i in range(1,N+1):
        for j in range(1,Request_list[i-1].a):
            rownames.append("rd_" + str(i)+ "_" + str(j) )
            rows.append([["x_"+str(i)+"_" + str(j)], [1]])
            senses = senses + "E"
            rhs.append(0)
        for j in range(Request_list[i-1].d-Request_list[i-1].e+2,T+1):
            rownames.append("rd_" + str(i)+ "_" + str(j) )
            rows.append([["x_"+str(i)+"_" + str(j)], [1]])
            senses = senses + "E"
            rhs.append(0)

    '''print(rownames)
    print(rows)
    print(rhs)
    print(obj)'''

    prob = cplex.Cplex()
    prob.set_log_stream(None)
    prob.set_error_stream(None)
    prob.set_warning_stream(None)
    prob.set_results_stream(None)
    prob.objective.set_sense(prob.objective.sense.maximize)

    prob.variables.add(obj=obj, ub=ub, lb=lb, types=types, names=varnames)
    prob.linear_constraints.add(lin_expr=rows, senses=senses, rhs=rhs, names=rownames)
    prob.solve()

    solution_x=prob.solution.get_values()[0:N*T]
    for i in range(len(solution_x)):
        solution_x[i]=int(solution_x[i])
    return prob.solution.get_objective_value(),solution_x

if __name__ == '__main__':
    N = 2
    K = 2
    P = 2
    R = 3

    Q = [
            [3, 2, 1]
        ] * K
    C = [
            [10, 20, 30]
        ] * R
    print(Q)
    print(C)

    Request_list = [Request(3, 8, [[1, 0]] * K, 2, 2),
                    Request(1, 9, [[1, 0, 1]] * K, 3, 4),
                    Request(2, 7, [[0, 1, 0]] * K, 3, 3), ]


    T = max([Request_list[i].d for i in range(N)])
    print([Request_list[i].d for i in range(N)])
    print(T)
    print(cplex_solve_question(N,K,P,R,Q,C,Request_list,T))
