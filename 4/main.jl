using JSON
import LinearAlgebra

function A(i::Int64, j::Int64)::Float64
  return 4 * i * j / (2 * (i + j) - 1)
end

function B(i::Int64, j::Int64)::Float64
  return 1 / (2 * (i + j) + 1) - 1 / (2 * i + 1) - 1 / (2 * j + 1) + 1
end

function C(i::Int64)::Float64
  return 1 - 1 / (2 * i + 1)
end

function calculate_lambda(a::LinearAlgebra.Symmetric{Float64}, b::LinearAlgebra.Symmetric{Float64})::Vector{Float64}
  @assert size(a) == size(b)
  L, U = LinearAlgebra.cholesky(b)
  c = LinearAlgebra.inv(L) * a * LinearAlgebra.inv(U)
  return LinearAlgebra.eigvals(c)
end

function calculate_alpha(a::LinearAlgebra.Symmetric{Float64}, b::LinearAlgebra.Symmetric{Float64}, lambda::Vector{Float64})::Matrix{Float64}
  @assert size(a) == size(b)
  @assert size(a) == (size(lambda)..., size(lambda)...)
  alpha = zeros(Float64, size(a))
  for (i, l) in enumerate(lambda)
    c = a - l * b
    eig = LinearAlgebra.eigen(c)
    @assert isapprox(eig.values[i], 0.0; atol=1e-12)
    alpha[i, :] = eig.vectors[:, i]
  end
  return alpha
end

function calculate_c(alpha::Matrix{Float64})::Vector{Float64}
  N, M = size(alpha)
  @assert N == M
  a = [
    sum( 
      alpha[k, l] * alpha[n, m] * B(l, m)
      for l in range(1, N), m in range(1, N)
    )
    for k in range(1, N), n in range(1, N)
  ]
  b = [sum(alpha[n, m] * C(m) for m in range(1, N)) for n in range(1, N)]
  return a \ b
end

function main()
  N = parse(Int64, ARGS[1])

  a = LinearAlgebra.Symmetric([A(i, j) for i in range(1, N), j in range(1, N)])
  b = LinearAlgebra.Symmetric([B(i, j) for i in range(1, N), j in range(1, N)])

  lambda = calculate_lambda(a, b)
  alpha = calculate_alpha(a, b, lambda)
  c = calculate_c(alpha)

  payload = Dict(
    "alpha" => alpha,
    "lambda" => lambda,
    "c" => c,
  )
  print(JSON.json(payload), "\n")
end

main()
