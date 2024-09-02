import numpy as np
import matplotlib.pyplot as plt
import copy

def apply_D(x: np.ndarray) -> np.ndarray:
    y = x[0:-1] - x[1:]
    return y

def apply_D_conjugate(y: np.ndarray) -> np.ndarray:
    x = np.zeros(len(y)+1)
    x[0:-1] = x[0:-1] + y
    x[1:] = x[1:] - y
    return x

def apply_D2(x: np.ndarray) -> np.ndarray:
    y = -x[0:-2] + 2*x[1:-1] - x[2:]
    return y

def apply_D2_conjugate(y: np.ndarray) -> np.ndarray:
    x = np.zeros(len(y)+2)
    x[0:-2] = x[0:-2] - y
    x[1:-1] = x[1:-1] + 2*y
    x[2:] = x[2:] - y
    return x


def ChambollePock(input, D, D_conjugate, Norm_D, l, k_Huber, min_error, max_iter):
    # input: data to be denoised
    # For parameters (D, D_conjugate, Norm_D) set eighter
    #   (apply_D2, apply_D2_conjugate, l>=4) or
    #   (apply_D,  apply_D_conjugate,  l>=2).
    # l: smoothing parameter.
    #   higher l will lead to smoother results.
    # k_Huber: Lipschitz constant for the Huber function.
    #   smaller k_Huber leads to large jumps (e.g. peaks) being relatively less penalized.
    # 
    
    sigma = 1/Norm_D
    tau = 1/Norm_D
    
    error = float('inf')
    iter = 0
    x = input
    y = D(input)
    xbar = np.copy(x)
    xprev = np.copy(x)
    while error > min_error and iter < max_iter:
        # Chabolle-Pock iterations
        y += sigma*D(xbar)
        y = y/(1 + sigma)
        y = np.divide(y, np.maximum(abs(y)/k_Huber, np.ones_like(y)))

        x -= tau*D_conjugate(y)
        x = (tau*input + l*x)/(tau + l)

        xbar = 2*x - xprev
        error = np.linalg.norm(x - xprev)

        xprev = np.copy(x)
        iter += 1

        if np.mod(iter, 100) == 0:
            print(error)
    print('finished after', iter, 'iterations')
    print(error)
    return x, y

if __name__ == "__main__":
    rng = np.random.default_rng()

    Nvert = 100
    Nedges = Nvert - 1

    input = rng.random(Nvert)

    x, y = ChambollePock(input, apply_D, apply_D_conjugate, 2, 1e1, 1e-1, 1e-10, 1e5)
    x2, y2 = ChambollePock(input, apply_D2, apply_D2_conjugate, 4, 2.5e1, 5e-1, 1e-10, 1e5)

    fig = plt.figure()
    ax = plt.subplot(111)
    line_input, = ax.plot(input)
    line_D1, = ax.plot(x)
    line_D2, = ax.plot(x2)

    leg = ax.legend([line_input, line_D1, line_D2], ['input', 'D_1', 'D_2'])

    plt.show()  