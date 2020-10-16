using BcdV3
using StaticArrays
using MimuConstants
using LinearAlgebra
using Printf


# Noise less simulations
TT = Float64
active_imus = [3,4,5,6, 27,28, 29, 30]
r_all = InitialPosMIMU4444BT()
r_m = r_all[:, active_imus]
r_ref = r_m[:,1]
r_m .-= r_ref
r_all .-= r_ref

sig_a = get_rms_noise_acc()
sig_g = get_rms_noise_gyro()
Na = size(r_m, 2)
Ng = 8

N = 40 # Dyn
Nt = Na + Ng

r = SMatrix{3,Na, TT}(r_m)

w = [deg2rad(1000.0) * @SVector randn(3) for _ = 1:N]
w_dot = [deg2rad(40e3) * @SVector randn(3) for _ = 1:N]
s = [20.0 * @SVector randn(3) for _ = 1:N]


u = vcat(
    [Ω(w[n])^2*r[:,k] + Ω(w_dot[n])*r[:,k] + s[n] for k = 1:Na, n = 1:N],
    [w[n] for k = range(Na+1,length=Ng), n = 1:N]
)
Qu_inv = [SMatrix{3,3,Float64}(I)/( k <= Na ? sig_a^2 :  sig_g^2) for k = 1:Nt]
w0 = [w[n] + 0.1*randn(3) for  n = 1:N]

tol_gs =  1.0e-13
i_max_w = 200
w_hat, w_dot_hat, s_hat =  dynamics(w0, u, Qu_inv, r, Ng, Val{3*(Na+Ng)}(),
    tol_gs, i_max_w)
