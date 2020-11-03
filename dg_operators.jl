include("Elements.jl");

# numerical fluxes
central(ρ⁻, ρ⁺) = (ρ⁻ + ρ⁺) / 2
right(ρ⁻, ρ⁺) = ρ⁺
left(ρ⁻, ρ⁺) = ρ⁻

# DG D4 operators
function D_DG(ρ, JI, LIFT, D, numflux)
  K = size(ρ, 2)
  # element derivative
  Dρ = JI * D * ρ

  # Face continuity
  ρ⁻ = @view ρ[end, :]
  ρ⁺ = @view ρ[ 1, mod1.(2:K+1, K)]
  Dρ[end, :] += LIFT * (numflux(ρ⁻, ρ⁺) - ρ⁻)

  ρ⁻ = @view ρ[1, :]
  ρ⁺ = @view ρ[end, mod1.(0:K-1, K)]
  Dρ[1, :] -= LIFT * (numflux(ρ⁺, ρ⁻) - ρ⁻)

  return Dρ
end

function dg_data(N, K, start::FT, stop::FT;
                 numflux = (central, central, central, central)) where FT
  elems = range(start, stop = stop, length=K + 1)

  # Grid spacing
  Δx = FT(elems.step)

  # LGL points and weights
  ξ, ω = lglpoints(FT, N)

  # Get the derivative matrix
  D = spectralderivative(ξ)

  # Create the degree of freedom mesh
  x = Δx * (ξ .+ 1) / 2 .+ elems[1:end-1]'

  # Inverse of the Jacobian determinant
  JI = 2 / Δx

  # lift operator
  LIFT = JI / ω[1]

  D4 = zeros(FT, (N+1) * K, (N+1) * K)

  v = zeros(FT, N+1, K)

  for n = 1:length(v)
    v[n] = 1
    Dv = D_DG(v, JI, LIFT, D, numflux[1])
    D2v = D_DG(Dv, JI, LIFT, D, numflux[2])
    D3v = D_DG(D2v, JI, LIFT, D, numflux[3])
    D4[:, n] = D_DG(D3v, JI, LIFT, D, numflux[4])[:]
    v[n] = 0
  end

  return (D4 = D4, x = x, M = ω  / JI)
end
