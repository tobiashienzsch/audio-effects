import matplotlib.pyplot as plt
import numpy as np
import sympy
from sympy.utilities.codegen import codegen
from sympy.utilities.lambdify import lambdify
from sympy.codegen.rewriting import create_expand_pow_optimization, optimize, optims_c99

sympy.init_printing(use_unicode=False, wrap_line=False)

x = sympy.Symbol('x', real=True)
T = sympy.Symbol('T', real=True)
R = sympy.Symbol('R', real=True)
W = sympy.Symbol('W', real=True)

# T = sympy.Rational(-24, 1)
# R = sympy.Rational(6, 1)
# W = sympy.Rational(18, 1)

# Hard-Knee
hard = sympy.Piecewise((x, sympy.Le(x, T)), (T+(x-T)/R, True))

# Soft-Knee
soft = sympy.Piecewise(
    (x, sympy.Lt(2*(x-T), -W)),
    (x+(1/R-1)*((x-T+W/2)**2)/(2*W), sympy.Le(2*sympy.Abs(x-T), W)),
    (T+(x-T)/R, True), #sympy.Gt(2*(x-T), W)
)
f = soft
# f = x+(1/R-1)*((x-T+W/2)**2)/(2*W)
AD1 = sympy.integrate(f, x)
AD2 = sympy.integrate(AD1, x)

expand_opt = create_expand_pow_optimization(5)
f_opt = optimize(expand_opt(sympy.simplify(f)), optims_c99)
# AD1_opt = optimize(expand_opt(sympy.simplify(AD1)), optims_c99)
# AD2_opt = optimize(expand_opt(sympy.simplify(AD2)), optims_c99)

[(c_name, c_code), (h_name, c_header)] = codegen(
    [("f", f)],
    "C99",
    header=False,
    empty=True
)

print(c_code)

# x_in = np.linspace(-36.0, 0.0, 256)
# hard_out = lambdify(x, hard, 'numpy')(x_in)
# soft_out = lambdify(x, soft, 'numpy')(x_in)

# plt.plot(x_in, hard_out, label="Hard-Knee")
# plt.plot(x_in, soft_out, label="Soft-Knee")
# plt.grid(which="major", linewidth=1)
# plt.grid(which="minor", linewidth=0.2)
# plt.minorticks_on()
# plt.legend()
# plt.show()
