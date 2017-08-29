# Macro free version of http://nbviewer.jupyter.org/github/xukai92/TuringDemo/blob/master/Differential%20Equation/turing-de.jl.ipynb
using DifferentialEquations
using ParameterizedFunctions
using RecursiveArrayTools
using Turing
using Plots
srand(31415926) # fix random seed just for demo purpose

f = @ode_def_nohes LotkaVolterraTest begin
    dx = a*x - b*x*y
    dy = -c*y + d*x*y
end a=>1.5 b=1.0 c=3.0 d=1.0

u0 = [1.0;1.0]
tspan = (0.0,10.0)

prob = ODEProblem(f,u0,tspan)
sol = solve(prob,Tsit5())
t = collect(linspace(0,10,40)) # I reduce the number of data so that I can see error bars from the 2nd prediction way
sig = 0.49  # nosie level
data = convert(Array, VectorOfArray([(sol(t[i]) + sig*randn(2)) for i in 1:length(t)]))

function problem_new_parameters(prob::ODEProblem,p)
  f = (t,u,du) -> prob.f(t,u,p,du)
  uEltype = eltype(p)
  u0 = [uEltype(prob.u0[i]) for i in 1:length(prob.u0)]
  tspan = (uEltype(prob.tspan[1]),uEltype(prob.tspan[2]))
  ODEProblem(f,u0,tspan)
end

# Force Turing.jl to initialize its compiler
# @model bi(x) = begin end
# bi(data)
# Above is not required if ~ notation is removed

bif(x=data; vi=Turing.VarInfo(), sampler=nothing) = begin

    # Define prior
    # a ~ Truncated(Normal(1.5, 1), 0.5, 2.5)  # DE param
    a = Turing.assume(sampler,
                      Truncated(Normal(1.5, 1), 0.5, 2.5),
                      Turing.VarName(vi, [:bif, :a], ""),
                      vi)

    # σ ~ InverseGamma(2, 3)                   # data noise
    σ = Turing.assume(sampler,
                      InverseGamma(2, 3),
                      Turing.VarName(vi, [:bif, :σ], ""),
                      vi)

    # Update solver
    p_tmp = problem_new_parameters(prob, a); sol_tmp = solve(p_tmp,Tsit5())

    # Observe data
    # Here you can do a lot of ways to write the observation
    # and a lot of possible optimization to make the program faster
    # E.g. naively

    for i = 1:length(t)
        res = sol_tmp(t[i])
        # x[:,i] ~ MvNormal(res, σ*ones(2))
        Turing.observe(
          sampler,
          MvNormal(res, σ*ones(2)),   # Distribution
          x[:,i],    # Data point
          vi
        )
    end

    vi
end

chn = sample(bif, HMC(1000, 0.02, 4))
