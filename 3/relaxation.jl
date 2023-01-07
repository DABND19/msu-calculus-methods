import JSON
import LinearAlgebra
import Profile

const TOL::Float64 = 1e-6
const MAX_ITERAIONS::Int64 = 100000

const H::Float64 = parse(Float64, ARGS[1])
const M_INFTY::Float64 = parse(Float64, ARGS[2])
const EPS::Float64 = parse(Float64, ARGS[3])
const N_X::Int64 = parse(Int64, ARGS[4])
const N_Y::Int64 = parse(Int64, ARGS[5])

const X_LEFT::Float64 = -10.0
const X_RIGHT::Float64 = 10.0

const Y_BOTTOM::Float64 = 0.0
const Y_TOP::Float64 = H

function f(x::Float64)::Float64
  if -1 <= x <= 1
    return EPS * (1 - x^2)
  end
  return 0
end

function init_psi!(psi::Matrix{Float64}, x::LinRange{Float64}, y::LinRange{Float64})
  fill!(psi, 0.0)
  psi[:, begin] = map(x -> -f(x), x)
end

function main()
  x = LinRange(X_LEFT, X_RIGHT, N_X + 1)
  h_x::Float64 = step(x)

  y = LinRange(Y_BOTTOM, Y_TOP, N_Y + 1)
  h_y::Float64 = step(y)

  A::Float64 = 1 - M_INFTY^2
  B::Float64 = (h_x^2) / (h_y^2)
  C::Float64 = -2 * (A + B)

  psi = zeros((size(x)..., size(y)...))
  init_psi!(psi, x, y)

  psi_next = zeros((size(x)..., size(y)...))
  init_psi!(psi_next, x, y)

  M = LinearAlgebra.SymTridiagonal(
    C * ones(size(psi, 1) - 2), B * ones(size(psi, 1) - 3)
  )
  ldltM = LinearAlgebra.ldlt(M)

  for it in 1:MAX_ITERAIONS
    init_psi!(psi_next, x, y)
    @inbounds for i in firstindex(psi, 1) + 1:lastindex(psi, 1) - 1
      s = -A * (psi_next[i - 1, firstindex(psi_next, 2) + 1:lastindex(psi_next, 2) - 1] + psi[i + 1, firstindex(psi, 2) + 1:lastindex(psi, 2) - 1])
      s[begin] -= B * psi[i, begin]
      s[end] -= B * psi[i, end]
      psi_next[i, firstindex(psi_next, 2) + 1:lastindex(psi_next, 2) - 1] = ldltM \ s
    end
    psi, psi_next = psi_next, psi

    norm = maximum(abs, psi - psi_next)
    print(stderr, "Iteration: $(it). Discrepancy: $(norm).\n")
    if norm <= TOL
      break
    end
  end

  payload = Dict(
    "psi" => psi,
    "x" => collect(x),
    "y" => collect(y),
    "M_INFTY" => M_INFTY,
    "EPS" => EPS,
  )
  print(JSON.json(payload), "\n")
end

main()
