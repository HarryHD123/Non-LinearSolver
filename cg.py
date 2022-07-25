from sympy import *
import time

# Create Variables
x1, x2, x3, x4, x5, x6, x7, x8 = x = symbols("x1 x2 x3 x4 x5 x6 x7 x8")

# --- USER INPUT ---
# Enter Function (use symbols x1, x2, x3, etc.):

function = (x1 - 1) ** 2 + (x2 - 2) ** 2 + (x3 - 3) ** 2  # Answer = 1,2,3
# function = x1 -x2 + 2 * x1 ** 2 + 2 * x1 * x2 + x2 ** 2 # Answer = -1,1.5
# function = x1 ** 2 - 2 * x1 * x2 + 3 * x2 ** 2 - 4 * x1  # Answer = 2,0
# function = x1 ** 2 - 2 * x1 + 3 * x2 ** 2 - 4 * x1  # Answer = 3,0
# function = (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2  # Rosenbrock Function, Answer = 1,1

# Enter Initial Guess - Default is all zeros, (Optional):
x_init = [0.5, 0.5, 0.5]

# Enter Type of Beta Calculation in speech marks - Default is 'PR', (Optional):
# beta_type = 'FR'
# beta_type = 'HS'

# -- NOTE: ALTERING THIS PARAMETERS CAN CAUSE THE PROGRAM TO BECOME UNSTABLE --

# Enter whether quick solve is on or off - The rounded values for the current point are input to check if the
# function is solved - Default is on, (Optional)
# quick_solve = False

# Quick Search Significant Figures. A lower value solves faster but utilises more rounding so could result in an error
# qs_sigfig should be between 1 and 15 and be an integer - Default is 2, (Optional):
# qs_sigfig = 2

# Stuck Strictness - How many iterations can give the same value which does not solve the function to the requested
# qs_sigfig before trying new parameters - Default is 5 when qs_sigfig is 2, 10 when qs_sigfig is 3, (Optional):
# Note, a strictness greater than max_iter will have no effect
# stuck_strictness = 5

# Enter max number of iterations before a different beta type is automatically used to try to solve the problem -
# Default is 100, (Optional):
max_iter = 65

# -- NOTE: ALTERING THESE PARAMETERS IS NOT RECOMMENDED --

# Golden Search Bounds:
gs_bound = [(1, -1), (100, 0), (100, -1), (10, -10)]

# Golden Search Timeout of 2 seconds set:
timeout = 2


# --- CALCULATION FUNCTIONS ---

def Golden_Section_Search(function, gs_bound, limit=0.0001, timeout=timeout):
    """Golden Section Search"""

    alpha = symbols("alpha")
    searching = True
    failed = False
    upper_bound = gs_bound[0]
    lower_bound = gs_bound[1]
    golden_ratio = (5 ** 0.5 - 1) / 2
    d = golden_ratio * (upper_bound - lower_bound)

    p1 = lower_bound + d
    p2 = upper_bound - d

    # Set start time
    start_time = time.time()

    while searching:

        cur_time = time.time()

        f_p1 = function.subs(alpha, p1)
        f_p2 = function.subs(alpha, p2)

        if f_p1 < f_p2:
            lower_bound = p2
            p2 = p1
            d = golden_ratio * (upper_bound - lower_bound)
            p1 = lower_bound + d

        elif f_p1 > f_p2:
            upper_bound = p1
            p1 = p2
            d = golden_ratio * (upper_bound - lower_bound)
            p2 = upper_bound - d

        elif abs((p1 - p2)) < limit:
            gold_value = float(f"{p1:.5f}")
            searching = False

        if cur_time - start_time > timeout:  # If timeout is hit breaks
            failed = True
            searching = False
            gold_value = 0
            break

    return gold_value, failed


def Calc_Gradient(function, variables, previous_point):
    """Determines the gradient with respect to a variable"""

    gradient = []
    for x in variables:
        f_diff = diff(function, x)
        for x2 in variables:
            f_diff = f_diff.subs(x2, previous_point[variables.index(x2)])
        gradient.append(f_diff)
    gradient = Matrix(gradient)

    return gradient


def Calc_SearchDirect(g_new, beta, d):
    """Determines the search direction"""

    d_new = -g_new + beta * d

    return d_new


def Calc_Alpha(function, previous_point, d, gs_bound):
    """Determines Alpha using a one-dimensional minimising routine - the Golden Section Search"""

    alpha = symbols("alpha")
    sub_eqs = previous_point + alpha * d
    alp_tuples = []
    for i in range(NumVar):
        alp_tuples.append((variables[i], sub_eqs[i]))
    alp_eq = function.subs(alp_tuples)
    alp, failed = Golden_Section_Search(alp_eq, gs_bound)

    return alp, failed


def Calc_Beta(g, g_new, type='PR'):
    """Determines Beta without using the Hessian, the Default method is Polak-Ribiere" \
    "Method can be chosen by entering HS, PR, FR for Hestenes-Stiefel, Polak-Ribiere or Fletcher-Reeves" \
    "Enter the gradient first, the new gradient second"""

    g_T = g.T
    g_new_T = g_new.T
    failed = False

    try:
        if type == 'PR':
            beta = (g_new_T * (g_new - g)) * (g_T * g).inv()  # Polak-Ribiere
        elif type == 'HS':
            beta = (g_new_T * (g_new - g)) * (g_T * (g_new - g)).inv()  # Hestenes-Stiefel
        elif type == 'FR':
            beta = (g_new_T * g_new) * (g_T * g).inv()  # Fletcher-Reeves
    except:  # Covers NonInvertibleMatrixError
        beta = [0]
        failed = True

    return beta[0], failed


# --- ADMIN FUNCTIONS ---

def write_file(Filename, Text, Mode):
    "Writes to the a file"

    # This structure allows a string, integer, float, list or list of lists to be written successfully
    with open(Filename, Mode) as f:
        if type(Text) is str or type(Text) is int or type(Text) is float:
            f.write(str(Text) + "\n")
        elif type(Text) is Mul:
            f.write(str(Text))
        elif type(Text) is list:
            for i in Text:
                if type(i) is str:
                    f.write(str(i) + "   \t")
                elif type(i) is list:
                    f.write("\n")
                    for j in i:
                        f.write(str("{:.2f}".format(j)) + " \t")  # Tableau always reads as 2.d.p
    f.close()


def create_log(function, iter_dict, Answer, beta_type, beta_swap):
    "Creates the log.txt and output.txt files"

    # Log.txt initiated
    write_file('log.txt', "This file shows the inner workings of the non-linear programming solver package.\n"
                          "Each step is recorded and displayed here.\n", 'w')

    # Output.txt initiated
    write_file('output.txt', "This file displays the answers found using non-linear programming solver package.\n", 'w')

    # Initial iteration logged
    iteration = 1
    write_file('log.txt', f"This is iteration {iteration}:\n", 'a')
    write_file('log.txt', f"The initial point is: {iter_dict[1][0]}", 'a')
    write_file('log.txt', f"The initial search directions for each variable are: {iter_dict[1][2]}", 'a')
    write_file('log.txt', f"The initial lambda/alpha is: {iter_dict[1][3]}", 'a')
    write_file('log.txt', f"The new point is: {iter_dict[1][1]}\n", 'a')

    # Beta Type Stated
    try:
        iter_dict[2]
        if beta_swap:
            write_file('log.txt', f"The problem could not be solved within the max number of iterations using the "
                                  f"selected beta calculation method, therefore:", 'a')

        if beta_type == 'PR':
            write_file('log.txt', f"Beta Values are calculated using the Polak-Ribiere method.", 'a')
        elif beta_type == 'HS':
            write_file('log.txt', f"Beta Values are calculated using the Hestenes-Stiefel method.", 'a')
        elif beta_type == 'FR':
            write_file('log.txt', f"Beta Values are calculated using the Fletcher-Reeves method.", 'a')
    except KeyError:
        pass

    # Subsequent iterations logged
    all_iter_logged = False
    while not all_iter_logged:
        iteration += 1
        try:
            iter_dict[iteration]
            write_file('log.txt', f"\nThis is iteration {iteration}:", 'a')
            write_file('log.txt', f"\nThis iteration starts from: {iter_dict[iteration][0]}", 'a')
            write_file('log.txt', f"The search directions for each variable are: {iter_dict[iteration][2]}", 'a')
            write_file('log.txt', f"Beta is: {iter_dict[iteration][4]}", 'a')
            write_file('log.txt', f"Alpha is: {iter_dict[iteration][3]}", 'a')
            write_file('log.txt', f"The new point is: {iter_dict[iteration][1]}", 'a')
        except KeyError:
            all_iter_logged = True

    # Answer logged to log.txt and output.txt
    write_file('log.txt', f"\nThe minimum of the function {function} is found at:", 'a')
    write_file('output.txt', f"The minimum of the function {function} is found at:", 'a')
    for i in range(NumVar):
        write_file('log.txt', f"x{i + 1} = {float(Answer[i]):.2f}", 'a')  # Answer is logged to 2.d.p
        write_file('output.txt', f"x{i + 1} = {float(Answer[i]):.2f}", 'a')  # Answer is logged to 2.d.p


def add_iter_dict(iter_dict, iteration, x, x_new, d, alp, beta=0):
    # Add variables to iteration dictionary
    x_dict = []
    x_new_dict = []
    d_dict = []
    for i in x:
        x_dict.append(float(f"{float(i):.2f}"))
    for j in x_new:
        x_new_dict.append(float(f"{float(j):.2f}"))
    for k in d:
        d_dict.append(float(f"{float(k):.2f}"))
    alp_dict = float(f"{float(alp):.2f}")
    beta_dict = float(f"{float(beta):.2f}")
    iter_dict[iteration] = (x_dict, x_new_dict, d_dict, alp_dict, beta_dict)

    return iter_dict


# --- SET UP ---

# If function is not set, provides an error message and requests a function is input
try:
    function
except NameError:
    print("Please enter a function.")
    exit(1)

# Calculate number of variables
vars = []
str_func = str(function)
for i in range(len(str_func)):
    if str_func[i].lower() == 'x':
        vars.append(str_func[i].lower() + str_func[i + 1].lower())
vars_unique = set(vars)
NumVar = len(vars_unique)

# If x_init is not set or is the wrong length, automatically sets the initial guess to all zeros
try:
    x_init
except NameError:
    x_init = [0] * NumVar

if (type(x_init) != list) or (len(x_init) != NumVar):
    x_init = [0] * NumVar

# If beta_type is not set or is not one of the 3 options, sets the beta_type to 'PR' (Polak-Ribiere)
try:
    beta_type
except NameError:
    beta_type = 'PR'

if not ((beta_type != 'PR') or (beta_type != 'HS') or (beta_type != 'FR')):
    beta_type = 'PR'

# If quick_solve is not set or is not boolean, sets it off
try:
    quick_solve
except NameError:
    quick_solve = True

if type(quick_solve) != bool:
    quick_solve = True

# If qs_sigfig is not set, is not in the range python supports or is not an integer, sets it to 2
try:
    qs_sigfig
except NameError:
    qs_sigfig = 2

if qs_sigfig != type(int):
    qs_sigfig = 2

if qs_sigfig not in range(1, 15):
    qs_sigfig = 2

# If stuck_strictness is not set or is not an integer, sets it to 5
try:
    stuck_strictness
except NameError:
    stuck_strictness = 5

if stuck_strictness != type(int):
    stuck_strictness = 5

# If max_iter is not set or is not an integer, sets it to 100
try:
    max_iter
except NameError:
    max_iter = 100

if max_iter != type(int):
    max_iter = 100

# Create Variables List
variables = []
for i in range(NumVar):
    variables.append(f"x{i + 1}")


# --- MAIN ---

def Main(function, x_init, beta_type='PR', quick_solve=quick_solve, qs_sigfig=qs_sigfig, max_iter=max_iter,
         gs_bound=gs_bound, stuck_strictness=stuck_strictness):
    "Main function to run non-linear solver"

    # Set inital parameters
    iter_dict = {}
    alpha = symbols("alpha")
    iteration = 1
    StopCrit = False
    Solved = False
    beta_swap = False
    stuck = False
    bound_count = 0
    x_init = Matrix(x_init)

    # Calculate gradient
    g_list_init = Calc_Gradient(function, variables, x_init)

    # Calculate initial search direction
    d_list_init = -g_list_init

    # Calculate Lambda
    lamda_val, failed = Calc_Alpha(function, x_init, d_list_init, gs_bound[bound_count])

    # Update position
    x_list = x_init + lamda_val * d_list_init

    # Add variables to iteration dictionary
    add_iter_dict(iter_dict, iteration, x_init, x_list, d_list_init, lamda_val)

    g_list = g_list_init
    d_list = d_list_init

    while not StopCrit:
        # Calculate new gradient of steepest descent
        g_new_list = Calc_Gradient(function, variables, x_list)

        # Calculate beta for this iteration
        beta, failed = Calc_Beta(g_list, g_new_list, beta_type)

        # Calculate the new search direction vector
        d_new_list = Calc_SearchDirect(g_new_list, beta, d_list)

        # Checks if problem is solved
        test_tuples = []
        for i in range(NumVar):
            test_tuples.append((variables[i], round(x_list[i], qs_sigfig)))
        test_eq = function.subs(test_tuples)
        if test_eq == 0:
            Solved = True

        if quick_solve and Solved:  # Quick solve means if the function is correct the program stops
            StopCrit = True
            break

        elif not all(g_new_list):  # if quick solve is off, program stops when all gradients equal 0
            StopCrit = True
            break

        iteration += 1  # Iteration is incremented

        # Calculate step size alpha
        alp, failed = Calc_Alpha(function, x_list, d_new_list, gs_bound[bound_count])

        # Update position
        x_new_list = x_list + alp * d_new_list

        # Update lists for next iteration
        g_list = g_new_list
        d_list = d_new_list

        # Add variables to iteration dictionary
        add_iter_dict(iter_dict, iteration, x_list, x_new_list, d_new_list, alp, beta)

        # Update point
        x_list = x_new_list

        # Detects if program is stuck at the wrong value and moves on to new searching criteria
        if quick_solve and iteration >= stuck_strictness:
            stuck_list = []
            for i in range(stuck_strictness):
                stuck_list.append(iter_dict[iteration - 1 * i][1])  # Checks the last number of points
                first = stuck_list[0]
            if all(x_val == first for x_val in stuck_list):  # If all values are equal, stuck is set to true
                stuck = True

        # If the time to find alpha exceeds the limit or a solution is not found within a certain number of
        # iterations, tries new boundaries
        if failed or iteration > max_iter or stuck:
            x_list = x_init + lamda_val * d_list_init
            g_list = g_list_init
            d_list = d_list_init
            iteration = 1
            iter_dict = {}
            stuck = False
            add_iter_dict(iter_dict, iteration, x_init, x_list, d_list_init, lamda_val)
            bound_count += 1

            # If all boundaries have been tried, swaps beta type
            try:
                gs_bound[bound_count]
            except IndexError:
                bound_count = 0
                beta_swap = True
                if beta_type == 'HS' or 'FR':
                    beta_type = 'PR'
                elif beta_type == 'PR':
                    beta_type = 'FR'

        print("Iteration:", iteration, "| Beta Type:", beta_type, "| GS Bounds:",
              gs_bound[bound_count])  # To track the number of iterations

    return iter_dict, x_list, beta_type, beta_swap


if __name__ == '__main__':
    iter_dict, Answer, beta_type, beta_swap = Main(function, x_init, beta_type, quick_solve, qs_sigfig,
                                                   max_iter)  # Non-linear problem solved
    create_log(function, iter_dict, Answer, beta_type, beta_swap)  # Problem logged
