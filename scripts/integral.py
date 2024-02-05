import matplotlib.pyplot as plt
import numpy as np
import sympy
from sympy.utilities.lambdify import lambdify

sympy.init_printing(use_unicode=False, wrap_line=False)

# x = sympy.Symbol('x', positive=True, nonzero=True)
x = sympy.Symbol('x', real=True)


tanh_clip = sympy.tanh(x)
sin_clip = -sympy.sin(sympy.pi*0.5*x)


degree = 3
one_over_degree = sympy.Rational(1, degree)
norm_factor = sympy.Rational(degree - 1, degree)
inv_norm = sympy.Rational(1, norm_factor)
y = x * norm_factor
soft3_clip = (y - (y**degree) * one_over_degree)*inv_norm

degree = 5
one_over_degree = sympy.Rational(1, degree)
norm_factor = sympy.Rational(degree - 1, degree)
inv_norm = sympy.Rational(1, norm_factor)
y = x * norm_factor
soft5_clip = (y - (y**degree) * one_over_degree)*inv_norm

alpha = sympy.Symbol('alpha')
beta = sympy.Symbol('beta')
alpha = 1.79  # sympy.Rational(179, 100)
beta = sympy.Rational(1, 5)
diode_clip = beta * (sympy.exp(alpha*x) - 1.0)


# x_in = np.linspace(-1.25, 1.25, 128)
# sin_out = lambdify(x, sin_clip, 'numpy')(x_in)
# tanh_out = lambdify(x, tanh_clip, 'numpy')(x_in)
# soft3_out = lambdify(x, soft3_clip, 'numpy')(x_in)
# soft5_out = lambdify(x, soft5_clip, 'numpy')(x_in)
# diode_out = lambdify(x, diode_clip, 'numpy')(x_in)


# plt.plot(x_in, tanh_out, label="tanh")
# plt.plot(x_in, sin_out, label="sin")
# plt.plot(x_in, soft3_out, label="soft3")
# plt.plot(x_in, soft5_out, label="soft5")
# plt.plot(x_in, diode_out, label="diode")
# plt.grid(which="major", linewidth=1)
# plt.grid(which="minor", linewidth=0.2)
# plt.minorticks_on()
# plt.legend()
# plt.show()


f = sympy.sign(x)*2*x
f = (3-(2-3*x)**2) / 3
f = 1
f = sympy.Piecewise(  # Hard-clip
    (-1 , sympy.Le(x, -1)),
    (1 , sympy.Ge(x, 1)),
    (x, True),
)
# f = sympy.Piecewise(  # Overdrive
#     (sympy.sign(x)*2*x, sympy.And(sympy.Ge(x, 0) , sympy.Le(x, sympy.Rational(1,3)))),
#     ((3-(2-3*x)**2) / 3, sympy.Gt(x, sympy.Rational(1,3)) & sympy.Le(x, sympy.Rational(2,3))),
#     (1, True)
# )
# f = sympy.sign(x)*(1-sympy.exp(-sympy.Abs(x)))
# f = sympy.Abs(x)  # Full-wave
# f = sympy.Piecewise((x, sympy.Gt(x, 0)), (sympy.Abs(x), True))  # Half-wave
AD1 = sympy.integrate(f, x)
AD2 = sympy.integrate(AD1, x)
print(f)
print(AD1)
print(AD2)
