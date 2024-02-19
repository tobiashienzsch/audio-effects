import matplotlib.pyplot as plt
import numpy as np
import sympy
from sympy.utilities.lambdify import lambdify
from sympy.utilities.codegen import codegen
from sympy.codegen.rewriting import create_expand_pow_optimization, optimize, optims_c99

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
    (-1, sympy.Le(x, -1)),
    (1, sympy.Ge(x, 1)),
    (x, True),
)
f = sympy.Piecewise(  # Overdrive
    (sympy.sign(x)*2*x, sympy.And(sympy.Ge(x, 0), sympy.Le(x, sympy.Rational(1, 3)))),
    ((3-(2-3*x)**2) / 3, sympy.Gt(x, sympy.Rational(1, 3))
     & sympy.Le(x, sympy.Rational(2, 3))),
    (1, True)
)
f = sympy.sign(x)*(1-sympy.exp(-sympy.Abs(x)))
f = sympy.Abs(x)  # Full-wave
f = sympy.Piecewise((x, sympy.Gt(x, 0)), (sympy.Abs(x), True))  # Half-wave
f = sympy.Piecewise((1-sympy.exp(x), sympy.Gt(x, 0)),
                    (-1+sympy.exp(x), True))  # Soft-Clip Exp
f = 16*x**5 - 20*x**3 + 5*x
f = 0.2 * (sympy.exp(1.79*x) - 1.0)


degree = sympy.Symbol("degree", integer=True, positive=True)
one_over_degree = 1/degree
norm_factor = (degree - 1) / degree
inv_norm = 1 / norm_factor

# degree = 27
# one_over_degree = sympy.Rational(1, degree)
# norm_factor = sympy.Rational(degree - 1, degree)
# inv_norm = sympy.Rational(1, norm_factor)

y = x * norm_factor
f = (y - (y**degree) * one_over_degree)*inv_norm
f = x**2
AD1 = sympy.integrate(f, x)
AD2 = sympy.integrate(AD1, x)

print(f)
print(AD1)
print(AD2)

expand_opt = create_expand_pow_optimization(5)
f_opt = optimize(expand_opt(sympy.simplify(f)), optims_c99)
AD1_opt = optimize(expand_opt(sympy.simplify(AD1)), optims_c99)
AD2_opt = optimize(expand_opt(sympy.simplify(AD2)), optims_c99)

# f_opt = f
# AD1_opt = AD1
# AD2_opt = AD2

# print(f)
# print(AD1)

name = f"softClip"
[(c_name, c_code), (h_name, c_header)] = codegen(
    [(f"{name}", f_opt), (f"{name}_AD1", AD1_opt), (f"{name}_AD2", AD2_opt)],
    "C99",
    header=True,
    empty=True,
    # prefix="pprod",
    project="PPROD",
)

print(c_code)
