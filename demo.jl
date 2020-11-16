import ClimateMachine.Mesh.Elements
import Printf: @sprintf
import LinearAlgebra: eigvals, norm

# Floating point type
FT = Float64

# number of elements
Ks = 2 .^ (1:5) * 5

# Which numerical fluxes to use (d1, d2, d3, d4)
# numflux = (:central, :central, :central, :central) # odd-even behavior
numflux = (:left, :right, :right, :left) # no odd-even behavior

# Final time for PDE simulation
# t_final = FT(1 // 1000)
t_final = FT(1 // 10)

# polynomial order
# for N = 1:7
for N = 1:5
  # storage for the error
  ε_D4 = zeros(FT, length(Ks))
  ε_pde = zeros(FT, length(Ks))
  ε_Qt = zeros(FT, length(Ks))
  ε_Qrhs = zeros(FT, length(Ks))

  # Loop over mesh resolutions
  for (iter, K) = enumerate(Ks)

    # element boundaries
    elems = range(FT(0), stop = FT(10) * π, length=K + 1)
    Δx = FT(elems.step)

    # LGL points and weights
    ξ, ω = Elements.lglpoints(BigFloat, N)

    # Get the derivative matrix
    D = Elements.spectralderivative(ξ)

    ξ = FT.(ξ)
    D = FT.(D)
    ω = FT.(ω)

    # Create the degree of freedom mesh
    x = Δx * (ξ .+ 1) / 2 .+ elems[1:end-1]'
    JI = 2 / Δx

    # Compute the DG derivative
    function D_DG(ρ, flux)
      # element derivative
      Dρ = JI * D * ρ

      # Face continuity
      Nq = N + 1
      if flux == :central # average
        Dρ[Nq, :] += JI * (ρ[ 1, mod1.(2:K+1, K)] - ρ[Nq, :]) / 2ω[Nq]
        Dρ[ 1, :] -= JI * (ρ[Nq, mod1.(0:K-1, K)] - ρ[ 1, :]) / 2ω[ 1]
      elseif flux == :right # look right for the value
        Dρ[Nq, :] += JI * (ρ[ 1, mod1.(2:K+1, K)] - ρ[Nq, :]) / ω[Nq]
      elseif flux == :left # look left for the value
        Dρ[ 1, :] -= JI * (ρ[Nq, mod1.(0:K-1, K)] - ρ[ 1, :]) / ω[ 1]
      end

      # Return derivative
      Dρ
    end

    #
    # Derivative test
    #

    # Model proble ∂_{x}^4 sin(x) = sin(x)
    ρ = sin.(x)

    # Compute the 4th DG derivative
    Dρ = D_DG(ρ, numflux[1])
    D2ρ = D_DG(Dρ, numflux[2])
    D3ρ = D_DG(D2ρ, numflux[3])
    D4ρ = D_DG(D3ρ, numflux[4])

    # Compute the L2-error
    ε_D4[iter] = sqrt(sum(Δx * ω .* (D4ρ - ρ).^2))

    #
    # PDE test
    #

    # Form D4 matrix (poorman style!) so we can use exponential time integration
    A = zeros((N+1) * K, (N+1) * K)
    v = fill!(similar(ρ), 0)
    for n = 1:length(v)
      v[n] = 1
      Dv = D_DG(v, numflux[1])
      D2v = D_DG(Dv, numflux[2])
      D3v = D_DG(D2v, numflux[3])
      A[:, n] = D_DG(D3v, numflux[4])[:]
      v[n] = 0
    end

    # If operator is stable min(real.(eigvals(A))) ≈ 0
    # @show extrema(real.(eigvals(A)))

    # Solve ρ_t + ρ_xxxx = 0 with exact soln ρ(x, t) = exp(-t) * sin(x)
    ρ = sin.(x)
    ρ_exact = ρ * exp(-t_final)
    ρrhs_exact = ρ_exact - ρ

    # solve ρ_t = -A * ρ -> exp(-A * t) * ρ0
    ρ_disc = reshape(exp(-t_final * A) * ρ[:], size(ρ))
    ρrhs_disc = ρ_disc - ρ

    # Compute the L2-error
    ε_pde[iter] = sqrt(sum(Δx * ω .* (ρ_disc - ρ_exact).^2))

    # relative error
    # ε_Qt[iter] = norm(ρ_disc-ρ_exact)/norm(ρ_exact)
    # ε_Qrhs[iter] = norm(ρrhs_disc-ρrhs_exact)/norm(ρrhs_exact)
    ε_Qt[iter] = sqrt(sum(Δx * ω .* (ρ_disc - ρ_exact).^2))/sqrt(sum(Δx * ω .* (ρ_exact).^2))
    ε_Qrhs[iter] = sqrt(sum(Δx * ω .* (ρrhs_disc - ρrhs_exact).^2))/sqrt( sum(Δx * ω .* (ρrhs_exact).^2))

  end

  # Compute the rates of covergence for derivative and pde approximation
  rate_D4 = (log2.(ε_D4[2:end]) - log2.(ε_D4[1:end-1])) ./
            (log2.(Ks[1:end-1]) - log2.(Ks[2:end]))
  rate_pde = (log2.(ε_pde[2:end]) - log2.(ε_pde[1:end-1])) ./
             (log2.(Ks[1:end-1]) - log2.(Ks[2:end]))

  # Create a nice(ish) output string
  str = @sprintf """
  polynomial order = %d
  """ N

  # First level has no rate
  str *= @sprintf """
  num elements = %3d
     d^4: rate (error) =       (%.3e)
     pde: rate (error) =       (%.3e)
  """ Ks[1] ε_D4[1] ε_pde[1]
  # str *= @sprintf """
  # num elements = %3d
  #    Qt relative error = %.3e
  #    Qt-Qinit relative error = %.3e
  # """ Ks[1] ε_Qt[1] ε_Qrhs[1]

  # add in higher level errors and rates
  for (K, r_D4, e_D4, r_pde, e_pde, r_Qt, r_Qrhs) in zip(Ks[2:end], rate_D4, ε_D4[2:end], rate_pde, ε_pde[2:end], ε_Qt[2:end], ε_Qrhs[2:end])
    str *= @sprintf """
    num elements = %3d
       d^4: rate (error) = %+.2f (%.3e)
       pde: rate (error) = %+.2f (%.3e)
    """ K r_D4 e_D4 r_pde e_pde
    # str *= @sprintf """
    # num elements = %3d
    #    Qt relative error = %.3e
    #    Qt-Qinit relative error = %.3e
    # """ K r_Qt r_Qrhs
  end
  @info str
  println()


end
