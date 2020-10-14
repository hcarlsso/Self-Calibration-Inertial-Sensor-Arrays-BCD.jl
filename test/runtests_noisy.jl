using Test

using BcdV3
using StaticArrays
using MimuConstants
using LinearAlgebra

# Ω(x) = BcdV3.Ω(x)
# Noise less simulations
TT = Float64
active_imus = [3,4,5,6, 27,28, 29, 30]
r_all = InitialPosMIMU4444BT()
r_m = r_all[:, active_imus]
r_ref = r_m[:,1]
r_m .-= r_ref
r_all .-= r_ref

g_m = get_normals_icosahedron()
N_orientions = size(g_m, 2)


sig_a = get_rms_noise_acc()
sig_g = get_rms_noise_gyro()
Na = size(r_m, 2)
Ng = 8

N = 40 # Dyn
N_per_orientation = Int(N/5)
Ns = ones(Int, N_orientions).*N_per_orientation
show_every = 100
N_bcd  = 400_000
tol_dlogp = 1.0e-9
pool = [1]
# pool = workers()
Nt = Na + Ng
g_mag = 9.81


g = [SVector{3,TT}(g_m[:,k]) for k = 1:N_orientions] .|> x -> x/norm(x)*g_mag

b = [@SVector randn(3) for k = 1:Nt]
T = [I + 1.0e-3*@SMatrix randn(3,3) for k = 1:Nt]
T[1] = I + 1.0e-3*SMatrix{3,3, Float64}(triu(randn(3,3)))


r = SMatrix{3,Na, TT}(r_m)

w = [deg2rad(1000.0) * @SVector randn(3) for _ = 1:N]
w_dot = [deg2rad(40e3) * @SVector randn(3) for _ = 1:N]
s = [20.0 * @SVector randn(3) for _ = 1:N]

y_s_true = vcat(
    [T[k]*(-g[n]) + b[k] for k = 1:Na, n = 1:N_orientions],
    [b[k] for k = range(Na+1,length=Ng), n = 1:N_orientions],
)
y_d_true = vcat(
    [T[k]*(Ω(w[n])^2*r[:,k] + Ω(w_dot[n])*r[:,k] + s[n]) + b[k] for k = 1:Na, n = 1:N],
    [T[k]*w[n]+b[k] for k = range(Na+1,length=Ng), n = 1:N]
)

y_s = [y_s_true[k,n] + 0.0*(k <= Na ? sig_a : sig_g)*randn(3)/sqrt(N_per_orientation) for k = 1:Nt, n = 1:N_orientions]
y_d = [y_d_true[k,n] + 0.0*( k <= Na ? sig_a : sig_g)*randn(3) for k = 1:Nt, n = 1:N]

Q_inv = [SMatrix{3,3,Float64}(I)/( k <= Na ? sig_a^2 :  sig_g^2) for k = 1:Nt]

b0 = [b_n + 0.0e-2*randn(3) for  b_n in b]
T0 = [T[k] + 0.0e-3*(k == 1 ? triu(randn(3,3)) : randn(3,3))  for k = 1:Nt]
r0_m = copy(r_m)
r0_m[:,2:end] += 0.0e-3*randn(3,Na-1)
r0 = SMatrix{3,Na}(r0_m)

println("BCD v2")
r_hat, T_hat, b_hat = @time bcd(
    r0, T0, b0, y_s, y_d, Q_inv, Ns, Na , Ng, g_mag, tol_dlogp, N_bcd,
    1.0e-12, 100, 1.0e-13, 200, pool;
    # callback = callback
    callback = get_callback_maxdiff_param(r, T, b, show_every) #callback
)
println("True")
display(r |> Array)
println("Hat")
display(r_hat |> Array)
println("r0")
display(r0 |> Array)
println("diff hat true")
display(r_hat - r |> Array .|> abs)
println("diff hat 0")
display(r_hat - r0 |> Array .|> abs)
println("sse r_hat")
display(sum((r_hat - r).^2))
println("sse r0")
display(sum((r0 - r).^2))
