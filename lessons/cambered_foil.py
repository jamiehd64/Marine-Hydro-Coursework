import VortexPanel as vp
import matplotlib.pyplot as plt
import numpy as np

def naca_offset(x,t): 
    """
    This function returns y_t for each value of x for a non-cambered NACA
    section. y_t is the offset from the centreline of the section.
    """
    return 5*t*(0.2969*np.sqrt(x)-0.1260*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)


c = 1 # chord


def mean_camber_line(x, m, p, c=c):
    """
    Defining the mean camber line. The function returns y_c, where only one
    of the two equations will be used depending on the value of x, as defined
    in the conditions
    """
    condition_1 = m/p**2 * (2*p*(x/c)-(x/c)**2)
    condition_2 = m/(1-p)**2 * ((1-2*p)+2*p*(x/c)-(x/c)**2)
    return np.where((x>=0)&(x<=(c*p)), condition_1, condition_2)


def dyc_over_dx(x, m, p, c=c):
    """
    defining the change of y in x on the mean camber line
    """
    condition_1 = 2*m/p**2 *(p - (x/c))
    condition_2 = 2*m/(1-p)**2 * (p - (x/c))
    return np.where((x>=0)&(x<=(c*p)),condition_1,condition_2)


def naca_final(x, m, p, t, c):
    dyc_dx = dyc_over_dx(x, m, p, c=c)
    th = np.arctan(dyc_dx)
    yt = naca_offset(x,t)
    yc = mean_camber_line(x, m, p, c=c)
    upper_surface = x - yt*np.sin(th), yc + yt*np.cos(th)
    lower_surface = x + yt*np.sin(th), yc - yt*np.cos(th)
    return (upper_surface, lower_surface)

N = 32
m = 0.03
p = 0.4
t = 0.15

theta = np.linspace(0,-2.*np.pi,N+1)
x = (np.cos(theta)+1)/2.
y = naca_offset(x,t=t)*np.sign(np.sin(theta))

def plot_naca(x, m, p, t, N):
        for i in naca_final(x, m, p, t, c):
            plt.plot(i[0], i[1],'r')
            plt.axis('equal')

plot_naca(x, m=m, p=p, t=t, N=N)

# velocity field around any section 

upper_surface = naca_final(x, m, p, t, c)[0]
lower_surface = naca_final(x, m, p, t, c)[1]

xs = np.concatenate((lower_surface[0][:-1], upper_surface[0]))
ys = np.concatenate((lower_surface[1][:-1], upper_surface[1]))
foil = vp.make_spline(N=N,x=xs,y=ys)   # define geometry
alpha = np.pi/16   # angle of attack
foil.solve_gamma(alpha,kutta=[(0,-1)])  # solve
foil.plot_flow()            # plot 
