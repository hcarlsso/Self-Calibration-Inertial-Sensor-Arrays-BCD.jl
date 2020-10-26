using BcdV3
using StaticArrays
using MimuConstants
using LinearAlgebra
using Printf
using Dates


# Noise less simulations
TT = Float64
active_imus = [3,4,5,6, 27,28, 29, 30]
r_all = get_InitialPosMIMU4444BT_corrected()
r_m = r_all[:, active_imus]
r_ref = r_m[:,1]
r_m .-= r_ref
r_all .-= r_ref

sig_a = get_rms_noise_acc()
sig_g = get_rms_noise_gyro()
Na = size(r_m, 2)
Ng = 8

N = 5000 # Dyn
Ns = []
show_every = 100
N_bcd  = 1000
tol_dlogp = 1.0e-9

Nt = Na + Ng
g_mag = 9.81

b = [@SVector randn(3) for k = 1:Nt]
T = [I + 1.0e-3*@SMatrix randn(3,3) for k = 1:Nt]
T[1] = I + 1.0e-3*SMatrix{3,3, Float64}(triu(randn(3,3)))
r = [SVector{3,TT}(r_m[:,k]) for k = 1:Na]

w = [deg2rad(1000.0) * @SVector randn(3) for _ = 1:N]
w_dot = [deg2rad(40e3) * @SVector randn(3) for _ = 1:N]
s = [20.0 * @SVector randn(3) for _ = 1:N]
w_m = hcat(Array.(w)...)

y_d_true = Array{SVector{3,TT}}(undef, Nt,N)
y_d_true_2 = Array{SVector{3,TT}}(undef, Nt,N)
for n = 1:N
    for k = 1:Na
        y_d_true[k,n] = T[k]*(Ω(w[n])^2*r[k] + Ω(w_dot[n])*r[k] + s[n]) + b[k]
        y_d_true_2[k,n] = T[k]*(Ω(2*w[n])^2*r[k] + Ω(2*w_dot[n])*r[k] + 2*s[n]) + b[k]
    end
    for k in range(Na+1, length=Ng)
        y_d_true[k,n] = T[k]*w[n] + b[k]
        y_d_true_2[k,n] = T[k]*w[n]*2 + b[k]
    end
end

Q_inv = [SMatrix{3,3,Float64}(I)/( k <= Na ? sig_a^2 :  sig_g^2) for k = 1:Nt]

b0 = [b_n + 1.0e-2*randn(3) for  b_n in b]
T0 = [T[k] + 1.0e-3*(k == 1 ? triu(randn(3,3)) : randn(3,3))  for k = 1:Nt]
r0_m = copy(r_m)
r0_m[:,2:end] += 1.0e-3*randn(3,Na-1)
r0 = [SVector{3}(r0_m[:,k]) for k = 1:Na]


y = Array{SVector{3,TT}}(undef, Nt,N)
y2 = Array{SVector{3,TT}}(undef, Nt,N)
for n = 1:N
    for k = 1:Na
        e = sig_a * randn(3)
        y[k,n] = y_d_true[k,n] + e
        y2[k,n] = y_d_true_2[k,n] + e
    end
    for k in range(Na+1, length=Ng)
        e = sig_g * randn(3)
        y[k,n] = y_d_true[k,n] + e
        y2[k,n] = y_d_true_2[k,n] + e
    end
end
dw = zeros(3, N)
dw2 = zeros(3, N)

w0 = deepcopy(w)
r_hat, T_hat, b_hat, eta = bcd!(
    copy(r), copy(T), copy(b), y, Q_inv, Ns, g_mag, Val{3*(Na+Ng)}(),Val{Na}(),
    1.0e-12, 100, 1.0e-8, 200, tol_dlogp, 1.0-14, 1; warn = false, w0 = w0
)
w_hat_m = hcat(Array.(eta[:w])...)
dw .+= (w_m  - w_hat_m).^2
mse1 = sum(dw)/length(dw)
rmse1 = sqrt(mse1)

r_hat, T_hat, b_hat, eta = bcd!(
    copy(r), copy(T), copy(b), y2, Q_inv, Ns, g_mag, Val{3*(Na+Ng)}(),Val{Na}(),
    1.0e-12, 100, 1.0e-8, 200, tol_dlogp, 1.0-14, 1; warn = false, w0 = w0.*2
)
w_hat_m = hcat(Array.(eta[:w])...)
dw2 .+= (w_m.*2  - w_hat_m).^2
mse2 = sum(dw2)/length(dw2)
rmse2 = sqrt(mse2)

# rmse is lower for second
