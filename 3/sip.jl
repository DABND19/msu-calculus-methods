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
const ALPHA::Float64 = parse(Float64, get(ARGS, 6, 0.0))

const X_LEFT::Float64 = -10.0
const X_RIGHT::Float64 = 10.0

const Y_BOTTOM::Float64 = 0.0
const Y_TOP::Float64 = H

struct PentadiagonalMatrix <: LinearAlgebra.AbstractMatrix{Float64}
  E::Vector{Float64}
  F::Vector{Float64}
  D::Vector{Float64}
  B::Vector{Float64}
  H::Vector{Float64}

  N::Int64

  function PentadiagonalMatrix(
    dll::Vector{Float64},
    dl::Vector{Float64},
    d::Vector{Float64},
    du::Vector{Float64},
    duu::Vector{Float64},
    N::Int64
  )
    return new(d, du, dl, dll, duu, N)
  end
end

function Base.size(A::PentadiagonalMatrix)::Tuple{Int64,Int64}
  return (size(A.E)..., size(A.E)...)
end

struct LowerTriangularTridiagonalMatrix <: LinearAlgebra.AbstractMatrix{Float64}
  d::Vector{Float64}
  c::Vector{Float64}
  b::Vector{Float64}
  N::Int64

  function LowerTriangularTridiagonalMatrix(
    d::Vector{Float64},
    dl::Vector{Float64},
    dll::Vector{Float64},
    ll_shift::Int64
  )
    return new(d, dl, dll, ll_shift)
  end
end

struct UnitUpperTriangularTridiagonalMatrix <: LinearAlgebra.AbstractMatrix{Float64}
  e::Vector{Float64}
  f::Vector{Float64}
  N::Int64

  function UnitUpperTriangularTridiagonalMatrix(
    du::Vector{Float64},
    duu::Vector{Float64},
    uu_shift::Int64
  )
    return new(du, duu, uu_shift)
  end
end

struct SemidiagonalMatrixLUFactorization <: LinearAlgebra.Factorization{Float64}
  L::LowerTriangularTridiagonalMatrix
  U::UnitUpperTriangularTridiagonalMatrix
end

function LinearAlgebra.factorize(A::PentadiagonalMatrix)::SemidiagonalMatrixLUFactorization
  (dim, _) = size(A)

  d = zeros(Float64, dim)
  e = zeros(Float64, dim)
  c = zeros(Float64, dim)
  b = zeros(Float64, dim)
  f = zeros(Float64, dim)

  d[begin] = A.E[begin]
  e[begin] = A.F[begin] / d[begin]
  f[begin] = A.H[begin] / d[begin]

  for i in 2:dim
    if i > A.N
      b[i] = A.B[i] / (1 + ALPHA * e[i-A.N])
    end

    c[i] = A.D[i] / (1 + ALPHA * f[i-1])

    if i > A.N
      d[i] = (
        A.E[i] + ALPHA * (b[i] * e[i-A.N] + c[i] * f[i-1])
        -
        b[i] * f[i-A.N] - c[i] * e[i-1]
      )
    else
      d[i] = A.E[i] + ALPHA * c[i] * f[i-1] - c[i] * e[i-1]
    end

    f[i] = (A.H[i] - ALPHA * c[i] * f[i-1]) / d[i]

    if i > A.N
      e[i] = (A.F[i] - ALPHA * b[i] * e[i-A.N]) / d[i]
    else
      e[i] = A.F[i] / d[i]
    end
  end

  L = LowerTriangularTridiagonalMatrix(d, c, b, A.N)
  U = UnitUpperTriangularTridiagonalMatrix(e, f, A.N)

  return SemidiagonalMatrixLUFactorization(L, U)
end

function Base.:*(A::PentadiagonalMatrix, psi::Vector{Float64})::Vector{Float64}
  res = zeros(size(psi))
  @simd for i in eachindex(res)
    @inbounds res[i] = A.E[i] * psi[i]

    if checkbounds(Bool, psi, i - 1)
      @inbounds res[i] += A.D[i] * psi[i-1]
    end

    if checkbounds(Bool, psi, i - A.N)
      @inbounds res[i] += A.B[i] * psi[i-A.N]
    end

    if checkbounds(Bool, psi, i + 1)
      @inbounds res[i] += A.F[i] * psi[i+1]
    end

    if checkbounds(Bool, psi, i + A.N)
      @inbounds res[i] += A.H[i] * psi[i+A.N]
    end
  end
  return res
end

function Base.:\(L::LowerTriangularTridiagonalMatrix, r::Vector{Float64})::Vector{Float64}
  v = zeros(size(r))
  for i in eachindex(v)
    @inbounds tmp::Float64 = r[i]

    if checkbounds(Bool, v, i - 1)
      @inbounds tmp -= L.c[i] * v[i-1]
    end

    if checkbounds(Bool, v, i - L.N)
      @inbounds tmp -= L.b[i] * v[i-L.N]
    end

    @inbounds v[i] = tmp / L.d[i]
  end
  return v
end

function Base.:\(U::UnitUpperTriangularTridiagonalMatrix, v::Vector{Float64})::Vector{Float64}
  delta = zeros(size(v))
  for i in reverse(eachindex(delta))
    @inbounds tmp::Float64 = v[i]

    if checkbounds(Bool, delta, i + 1)
      @inbounds tmp -= U.e[i] * delta[i+1]
    end

    if checkbounds(Bool, delta, i + U.N)
      @inbounds tmp -= U.f[i] * delta[i+U.N]
    end

    @inbounds delta[i] = tmp
  end
  return delta
end

function f(x::Float64)::Float64
  if -1 <= x <= 1
    return EPS * (1 - x^2)
  end
  return 0
end

function main()
  x = LinRange(X_LEFT, X_RIGHT, N_X)
  y = LinRange(Y_BOTTOM, Y_TOP, N_Y)
  (h_x, h_y) = step(x), step(y)

  psi = zeros(N_X, N_Y)
  psi[:, begin] = map(x -> -f(x), x)

  q = zeros(N_X, N_Y)
  q[:, begin] = psi[:, begin]

  c_du = c_dl = -(1 - M_INFTY^2)
  c_duu = c_dll = -(h_x / h_y)^2
  c_d = -2.0 * (c_du + c_duu)

  dll = c_dll * ones(size(psi))
  dl = c_dl * ones(size(psi))
  d = c_d * ones(size(psi))
  du = c_du * ones(size(psi))
  duu = c_duu * ones(size(psi))

  d[begin, :] .= 1.0
  d[end, :] .= 1.0
  d[:, begin] .= 1.0
  d[:, end] .= 1.0

  for m in (dll, dl, du, duu)
    m[begin, :] .= 0.0
    m[end, :] .= 0.0
    m[:, begin] .= 0.0
    m[:, end] .= 0.0
  end

  psi = vec(psi)
  q = vec(q)
  d = vec(d)
  dl = vec(dl)
  du = vec(du)
  duu = vec(duu)
  dll = vec(dll)

  A = PentadiagonalMatrix(dll, dl, d, du, duu, N_X)
  LU::SemidiagonalMatrixLUFactorization = LinearAlgebra.factorize(A)

  for it in Base.OneTo(MAX_ITERAIONS)
    r = q - A * psi
    v = LU.L \ r
    delta = LU.U \ v
    norm = maximum(abs, delta)
    if norm <= TOL
      break
    end
    print(stderr, "Iteration: $(it). Discrepancy: $(norm).\n")
    psi += delta
  end

  payload = Dict(
    "psi" => reshape(psi, N_X, N_Y),
    "x" => collect(x),
    "y" => collect(y),
    "M_INFTY" => M_INFTY,
    "EPS" => EPS,
  )
  print(JSON.json(payload), "\n")
end

main()
