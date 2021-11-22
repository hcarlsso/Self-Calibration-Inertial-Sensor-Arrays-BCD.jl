using BcdV3
using StaticArrays
using MimuConstants
using LinearAlgebra
using Printf
using Dates
using Random
Random.seed!(1)

function InitialPosMIMU4444BT()
    dx = 0.0063; dy = 0.0063; dz = 0.001;
    r = [-1.5*dx  0.5*dy  1*dz;   #  1
         -1.5*dx  0.5*dy -1*dz;   #  2
         -1.5*dx  1.5*dy  1*dz;   #  3
         -1.5*dx  1.5*dy -1*dz;   #  4
         -1.5*dx -1.5*dy  1*dz;   #  5
         -1.5*dx -1.5*dy -1*dz;   #  6
         -1.5*dx -0.5*dy  1*dz;   #  7
         -1.5*dx -0.5*dy -1*dz;   #  8
         -0.5*dx  0.5*dy  1*dz;   #  9
         -0.5*dx  0.5*dy -1*dz;   # 10
         -0.5*dx  1.5*dy  1*dz;   # 11
         -0.5*dx  1.5*dy -1*dz;   # 12
         -0.5*dx -1.5*dy  1*dz;   # 13
         -0.5*dx -1.5*dy -1*dz;   # 14
         -0.5*dx -0.5*dy  1*dz;   # 15
         -0.5*dx -0.5*dy -1*dz;   # 16
          0.5*dx  0.5*dy  1*dz;   # 17
          0.5*dx  0.5*dy -1*dz;   # 18
          0.5*dx  1.5*dy  1*dz;   # 19
          0.5*dx  1.5*dy -1*dz;   # 20
          0.5*dx -1.5*dy  1*dz;   # 21
          0.5*dx -1.5*dy -1*dz;   # 22
          0.5*dx -0.5*dy  1*dz;   # 23
          0.5*dx -0.5*dy -1*dz;   # 24
          1.5*dx  0.5*dy  1*dz;   # 25
          1.5*dx  0.5*dy -1*dz;   # 26
          1.5*dx  1.5*dy  1*dz;   # 27
          1.5*dx  1.5*dy -1*dz;   # 28
          1.5*dx -1.5*dy  1*dz;   # 29
          1.5*dx -1.5*dy -1*dz;   # 30
          1.5*dx -0.5*dy  1*dz;   # 31
          1.5*dx -0.5*dy -1*dz]'; # 32
    return r
end

function get_normals_icosahedron()
    f = [
        0.1876 -0.7947 0.5774
        0.6071 -0.7947 0.0000
        -0.4911 -0.7947 0.3568
        -0.4911 -0.7947 -0.3568
        0.1876 -0.7947 -0.5774
        0.9822 -0.1876 0.0000
        0.3035 -0.1876 0.9342
        -0.7946 -0.1876 0.5774
        -0.7946 -0.1876 -0.5774
        0.3035 -0.1876 -0.9342
        0.7946 0.1876 0.5774
        -0.3035 0.1876 0.9342
        -0.9822 0.1876 0.0000
        -0.3035 0.1876 -0.9342
        0.7946 0.1876 -0.5774
        0.4911 0.7947 0.3568
        -0.1876 0.7947 0.5774
        -0.6071 0.7947 0.0000
        -0.1876 0.7947 -0.5774
        0.4911 0.7947 -0.3568
    ]
    f_norm = f ./ sqrt.(sum(f.^2, dims = 2))
    return f_norm'
end

function get_callback_trace(trace, trace_p, show_every = 1)
    function callback(k::Int, dx, dlogp, log_p, r, T, b)
        trace[:f][k] = log_p
        trace[:f_abs][k] = dlogp
        trace[:f_rel][k] = abs(dlogp)/abs(log_p)

        trace[:x_abs][k] = dx

        if mod(k,show_every) == 0
            time = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
            @printf("%20s %7d logp %.16e dlogp % .8e ||dx|| %.8e\n",
                    time, k, log_p, dlogp, trace[:x_abs][k])

            # C = 0.0
            # for k = 1:8
            #     C += tr(T[k+8])
            # end
            # C /= (3*8)
            # @printf(" C_norm %.8e\n", C)
            push!(trace_p[:r], Array(r))
            push!(trace_p[:T], Array.(T))
            push!(trace_p[:b], Array.(b))
            push!(trace_p[:k], k)
        end
    end
    return callback
end

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
show_every = 1
N_bcd  = 3
tol_dlogp = 1.0e-9

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

# y_s = [y_s_true[k,n] + 0.0*(k <= Na ? sig_a : sig_g)*randn(3)/sqrt(N_per_orientation) for k = 1:Nt, n = 1:N_orientions]
# y_d = [y_d_true[k,n] + 0.0*( k <= Na ? sig_a : sig_g)*randn(3) for k = 1:Nt, n = 1:N]
# y = hcat(y_s, y_d)

y = hcat(y_s_true, y_d_true)
Q_inv = [SMatrix{3,3,Float64}(I)/( k <= Na ? sig_a^2 :  sig_g^2) for k = 1:Nt]

b0 = [b_n + 1.0e-2*randn(3) for  b_n in b]
T0 = [T[k] + 1.0e-3*(k == 1 ? triu(randn(3,3)) : randn(3,3))  for k = 1:Nt]
r0_m = copy(r_m)
r0_m[:,2:end] += 1.0e-3*randn(3,Na-1)
r0 = [SVector{3}(r0_m[:,k]) for k = 1:Na]

trace = Dict(
    :f => zeros(N_bcd),
    :f_abs => zeros(N_bcd),
    :f_rel => zeros(N_bcd),
    :x_abs => zeros(N_bcd)
)
trace_p = Dict(:r => [], :T => [], :b => [], :k => [])
r_hat = copy(r0)
T_hat = copy(T0)
b_hat = copy(b0)

r_hat, T_hat, b_hat, eta = @time  bcd!(
    r_hat, T_hat, b_hat, y, Q_inv, Ns, g_mag, Val{3*(Na+Ng)}(),Val{Na}(),
    1.0e-12, 100, 1.0e-13, 200, tol_dlogp, 1.0-14, N_bcd;
    callback = get_callback_trace(
        trace, trace_p, show_every
    ), #callback
    verbose = false
);
