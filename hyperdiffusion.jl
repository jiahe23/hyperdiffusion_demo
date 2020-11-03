### A Pluto.jl notebook ###
# v0.12.6

using Markdown
using InteractiveUtils

# ╔═╡ ce2dfd4a-1deb-11eb-31f5-4d237b2d0610
include("dg_operators.jl");

# ╔═╡ 8e8eb794-1df0-11eb-3f3b-9f2c61ce3c3a
import Printf: @sprintf

# ╔═╡ 8e8f981c-1df0-11eb-2a06-c5686fc02af0
import LinearAlgebra: eigvals

# ╔═╡ d1a69f42-1df0-11eb-09b3-215314f63339
import Plots

# ╔═╡ 8e9477c4-1df0-11eb-188b-2ff136c41a1a
# First we define the computational parameters
begin
  # Floating Point Type
  FT = Float64

  # Polynomial Order
  N = 1

  # number of elements
  num_elems = 11

  # Final time for PDE simulation
  t_final = FT(1 // 10)

  # create the dt data for the simulation
  dg = dg_data(N, num_elems, FT(0), 2FT(π))
end;

# ╔═╡ 8ea3c5d0-1df0-11eb-0d3b-8bfedcbaa3ad
begin
  # First we test the function
  ρ = sin.(dg.x)

  # which has the property that ∂^4 ρ / ∂_x^4 = ρ
  D4ρ = reshape(dg.D4 * ρ[:], size(ρ))
end;

# ╔═╡ e083d63c-1df1-11eb-21c0-f5312717ad4d
begin
  # Plot the solution and the approximation... not so good!
  Plots.plot(dg.x, ρ, color = :red, legend = :none)
  Plots.plot!(dg.x, D4ρ, color = :blue, legend = :none)
end

# ╔═╡ c3740ae6-1df4-11eb-333f-a7ae14f8982e
begin
  # Looking at the error we see the obvious badnees :(
  Plots.plot(dg.x, ρ - D4ρ, color = :red, legend = :none)
end

# ╔═╡ 2ae63aa8-1df4-11eb-00d7-c328b731c322
begin
  # Let's instead evolve the PDE ρ_t = ρ_xxxx some small time
  # with ρ₀(x) = sin(x) the exact solution is ρ(x, t) = exp(-t) sin(x)
  ρ2 = reshape(exp(-t_final * dg.D4) * ρ[:], size(ρ))
  D4ρ2 = reshape(dg.D4 * ρ2[:], size(ρ2))
  D4ρ2_exact = exp(-t_final) * ρ
end;

# ╔═╡ f5a5924e-1df1-11eb-30e4-3908a1ac07cc
begin
  # Plot the solution and the approximation... looking better!
  Plots.plot(dg.x, ρ2, color = :red, legend = :none)
  Plots.plot!(dg.x, D4ρ2, color = :blue, legend = :none)
end

# ╔═╡ d278db02-1df4-11eb-0a90-d90e75af7ea3
begin
  # Looking at the error things are converging
  Plots.plot(dg.x, ρ2 - D4ρ2, color = :red, legend = :none)
end

# ╔═╡ fec8c72a-1e29-11eb-3ce4-af2701a644a7
# Now let's do a refinement study
let
  # These are the levels we will compute
  Ks = 2 .^ (1:8)

  # Loop over refinement levels and compute error
  err = zeros(FT, length(Ks))
  for (iter, K) in enumerate(Ks)
    dg = dg_data(N, K, FT(0), 2FT(π))
    ρ = sin.(dg.x)
    ρ2 = reshape(exp(-t_final * dg.D4) * ρ[:], size(ρ))
    err[iter] = sqrt(sum(dg.M .* (ρ2 - ρ * exp(-t_final)).^2))
  end

  # Compute the rates
  rate = (log2.(err[1:end-1]) - log2.(err[2:end])) ./
  (log2.(Ks[2:end])    - log2.(Ks[1:end-1]))

  # Make a nice ouput string
  str = @sprintf "```polynomial order = %d```\n" N
  str *= @sprintf "
  ```num elements = %03d :: rate (error) = XXXX (%.3e)```\n
  " Ks[1] err[1]
  for (K, e, r) in zip(Ks[2:end], err[2:end], rate)
    str *= @sprintf "```num elements = %03d :: rate (error) = %7.2f (%.3e)```\n\n" K r e
  end
  Markdown.parse(str)
end

# ╔═╡ Cell order:
# ╠═ce2dfd4a-1deb-11eb-31f5-4d237b2d0610
# ╠═8e8eb794-1df0-11eb-3f3b-9f2c61ce3c3a
# ╠═8e8f981c-1df0-11eb-2a06-c5686fc02af0
# ╠═d1a69f42-1df0-11eb-09b3-215314f63339
# ╠═8e9477c4-1df0-11eb-188b-2ff136c41a1a
# ╠═8ea3c5d0-1df0-11eb-0d3b-8bfedcbaa3ad
# ╠═e083d63c-1df1-11eb-21c0-f5312717ad4d
# ╠═c3740ae6-1df4-11eb-333f-a7ae14f8982e
# ╠═2ae63aa8-1df4-11eb-00d7-c328b731c322
# ╠═f5a5924e-1df1-11eb-30e4-3908a1ac07cc
# ╠═d278db02-1df4-11eb-0a90-d90e75af7ea3
# ╠═fec8c72a-1e29-11eb-3ce4-af2701a644a7
