from tkinter import *
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from pulp import *
import numpy as np
from sympy.solvers import solve
from sympy import Symbol

GUI = Tk()
GUI.title("Simplex Method")
GUI.geometry("500x500")

font1 = (None, 15)

f = Label(GUI, text="F = ", font=font1)
f.place(x=100, y=250)

coeff_xf = IntVar()
coeff_yf = IntVar()
equalf = IntVar()
ef = ttk.Entry(GUI, textvariable= coeff_xf, width=3, font=font1)
ef.place(x=200, y=265, anchor="center")
l1f = Label(GUI, text="X ", font= font1)
l1f.place(x=250, y=265, anchor="center")
l3f = Label(GUI, text=" + ", font= font1)
l3f.place(x=300, y=265, anchor="center")
e2f = ttk.Entry(GUI, textvariable= coeff_yf, width=3, font=font1)
e2f.place(x=350, y=265, anchor="center")
l2f = Label(GUI, text="Y ", font= font1)
l2f.place(x=400, y=265, anchor="center")
s = Label(GUI, text="-", font=font1)
s.place(x=410, y=250)
ss = Label(GUI, text=">", font=font1)
ss.place(x=420, y=252)
maximum = Label(GUI, text="max", font=font1)
maximum.place(x=450, y=250)


coeff_x = IntVar()
coeff_y = IntVar()
equal = IntVar()
e1 = ttk.Entry(GUI, textvariable= coeff_x, width=3, font=font1)
e1.place(x=100, y=100, anchor="center")
l1 = Label(GUI, text="X ", font= font1)
l1.place(x=150, y=100, anchor="center")
l3 = Label(GUI, text=" + ", font= font1)
l3.place(x=200, y=100, anchor="center")
e2 = ttk.Entry(GUI, textvariable= coeff_y, width=3, font=font1)
e2.place(x=250, y=100, anchor="center")
l2 = Label(GUI, text="Y ", font= font1)
l2.place(x=300, y=100, anchor="center")
l4 = Label(GUI, text=" <= ", font= font1)
l4.place(x=350, y=100, anchor="center")
e2 = ttk.Entry(GUI, textvariable= equal, width=4, font=font1)
e2.place(x=400, y=100, anchor="center")

# Equation2
coeff_x2 = IntVar()
coeff_y2 = IntVar()
equal2 = IntVar()
e2 = ttk.Entry(GUI, textvariable= coeff_x2, width=3, font=font1)
e2.place(x=100, y=150, anchor="center")
l12 = Label(GUI, text="X ", font= font1)
l12.place(x=150, y=150, anchor="center")
l32 = Label(GUI, text=" + ", font= font1)
l32.place(x=200, y=150, anchor="center")
e22 = ttk.Entry(GUI, textvariable= coeff_y2, width=3, font=font1)
e22.place(x=250, y=150, anchor="center")
l22 = Label(GUI, text="Y ", font= font1)
l22.place(x=300, y=150, anchor="center")
l42 = Label(GUI, text=" <= ", font= font1)
l42.place(x=350, y=150, anchor="center")
e22 = ttk.Entry(GUI, textvariable= equal2, width=4, font=font1)
e22.place(x=400, y=150, anchor="center")


coeff_x3 = IntVar()
coeff_y3 = IntVar()
equal3 = IntVar()
e3 = ttk.Entry(GUI, textvariable= coeff_x3, width=3, font=font1)
e3.place(x=100, y=200, anchor="center")
l13 = Label(GUI, text="X ", font= font1)
l13.place(x=150, y=200, anchor="center")
l33 = Label(GUI, text=" + ", font= font1)
l33.place(x=200, y=200, anchor="center")
e23 = ttk.Entry(GUI, textvariable= coeff_y3, width=3, font=font1)
e23.place(x=250, y=200, anchor="center")
l23 = Label(GUI, text="Y ", font= font1)
l23.place(x=300, y=200, anchor="center")
l43 = Label(GUI, text=" <= ", font= font1)
l43.place(x=350, y=200, anchor="center")
e23 = ttk.Entry(GUI, textvariable= equal3, width=4, font=font1)
e23.place(x=400, y=200, anchor="center")


def get():
    X = []
    Y = []
    x = coeff_x.get()
    X.append(x)
    y = coeff_y.get()
    Y.append(y)
    equal_result = equal.get()

    x2 = coeff_x2.get()
    X.append(x2)
    y2 = coeff_y2.get()
    Y.append(y2)
    equal_result2 = equal2.get()

    x3 = coeff_x3.get()
    X.append(x3)
    y3 = coeff_y3.get()
    Y.append(y3)
    equal_result3 = equal3.get()

    xf = coeff_xf.get()
    yf = coeff_yf.get()


    if x == "":
        messagebox.showwarning("Warning","Please fill in the Coefficient of x1!")
        return
    elif y == "":
        messagebox.showwarning("Warning","Please fill in the Coefficient of y1!")
        return
    elif x2 == "":
        messagebox.showwarning("Warning","Please fill in the Coefficient of x2!")
        return
    elif y2 == "":
        messagebox.showwarning("Warning","Please fill in the Coefficient of y2!")
        return
    elif x3 == "":
        messagebox.showwarning("Warning","Please fill in the Coefficient of x3!")
        return
    elif y3 == "":
        messagebox.showwarning("Warning","Please fill in the Coefficient of y3!")
        return
    elif xf == "":
        messagebox.showwarning("Warning","Please fill in the Coefficient of x3!")
        return
    elif yf == "":
        messagebox.showwarning("Warning","Please fill in the Coefficient of y3!")
        return
    elif equal_result == "":
        messagebox.showwarning("Warning","Please fill in the result of equation 1!")
        return
    elif equal_result2 == "":
        messagebox.showwarning("Warning","Please fill in the result of equation2!")
        return
    elif equal_result3 == "":
        messagebox.showwarning("Warning","Please fill in the result of equation23!")
        return

    coeff_x.set("")
    coeff_y.set("")
    equal.set("")

    coeff_x2.set("")
    coeff_y2.set("")
    equal2.set("")

    coeff_x3.set("")
    coeff_y3.set("")
    equal3.set("")

    coeff_xf.set("")
    coeff_yf.set("")

    x = int(x)
    y = int(y)
    equal_result = int(equal_result)

    x2 = int(x2)
    y2 = int(y2)
    equal_result2 = int(equal_result2)

    x3 = int(x3)
    y3 = int(y3)
    equal_result3 = int(equal_result3)

    xf = int(xf)
    yf = int(yf)



    # Create an object of a model
    prob = LpProblem("Simple LP Problem", LpMaximize)
    # Define the decision variables
    x_1 = LpVariable("x_1", 0)
    x_2 = LpVariable("x_2", 0)
    # Define the objective function
    prob += xf * x_1 + yf * x_2
    # Define the constraints
    prob += x * x_1 + y * x_2 <= equal_result, "1st constraint"
    prob += x2 * x_1 + y2 * x_2 <= equal_result2, "2nd constraint"
    prob += x3 * x_1 + y3 * x_2 <= equal_result3, "3rd constraint"
    # Solve the linear programming problem
    prob.solve()

    # Print the results
    print("Status: ", LpStatus[prob.status])

    vv=[]
    for v in prob.variables():
        vv.append(v.varValue)
        print(v.name, "=", v.varValue)

    print("The optimal value of the objective function is = ", value(prob.objective))

    # Plot the optimal solution

    mx=X[0]
    my=Y[0]
    for i in X:
        if mx<i:
            mx=i
    for i in Y:
        if my<i:
            my=i
    xx = np.arange(0, 100)
    equal_resultf = value(prob.objective)

    Fmax = Label(GUI, text="Fmax({},{}) = {}".format(vv[0], vv[1], equal_resultf), font=font1)
    Fmax.place(x=100, y=300)
    if y!=0:
        plt.plot(xx, equal_result / y - x * xx / y, label='{}x + {}y = {}'.format(x, y, equal_result))
    else:
        plt.axvline(x, label='{}x + {}y = {}'.format(x, y, equal_result))
    if y2!=0:
        plt.plot(xx, equal_result2 / y2 - x2 * xx / y2, label='{}x + {}y = {}'.format(x2, y2, equal_result2))
    else:
        plt.axvline(x2, label='{}x + {}y = {}'.format(x2, y2, equal_result2))
    if y3!=0:
        plt.plot(xx, equal_result3 / y3 - x3 * xx / y3, label='{}x + {}y = {}'.format(x3, y3, equal_result3))
    else:
        plt.axvline(x3, label='{}x + {}y = {}'.format(x3, y3, equal_result3))
    #plt.plot([0, xf], [0, yf], label="C")
    if xf!=0:
        plt.plot(xx, yf*xx/xf, label="C")
        plt.plot(xx, equal_resultf / yf - xf * xx / yf, '--', label="Fmax")
    else:
        plt.axvline(xf, label='C')
        plt.axvline(xf, label='Fmax')


    yy=np.arange(0, 100)

    if y<0:
        plt.fill_between(xx, equal_result / y - x * xx / y, np.max(equal_result / y - x * xx / y), color = 'blue', alpha = 0.5)
    elif y!=0:
        plt.fill_between(xx, equal_result / y - x * xx / y, color = 'blue', alpha = 0.5)
    else:
        plt.fill_betweenx(yy, x,color='blue', alpha=0.5)
    if y3<0:
        plt.fill_between(xx, equal_result3 / y3 - x3 * xx / y3, np.max(equal_result3 / y3 - x3 * xx / y3), color='green', alpha=0.5)
    elif y3!=0:
        plt.fill_between(xx, equal_result3 / y3 - x3 * xx / y3, color='green', alpha=0.5)
    else:
        plt.fill_betweenx(yy, x3, color='blue', alpha=0.5)
    if y2<0:
        plt.fill_between(xx, equal_result2 / y2 - x2 * xx / y2, np.max(equal_result2 / y2 - x2 * xx / y2), color='orange', alpha=0.5)
    elif y2!=0:
        plt.fill_between(xx, equal_result2 / y2 - x2 * xx / y2, color='orange', alpha=0.5)
    plt.fill_betweenx(yy, x2, color='blue', alpha=0.5)


    plt.axis([0, mx+10, 0, my+10])
    plt.grid(True)
    plt.legend()
    plt.show()

b1 = ttk.Button(GUI, text="Solve", command= get)
b1.place(x=250, y=350, anchor="center")

GUI.mainloop()

