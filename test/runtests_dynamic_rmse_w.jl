using BcdV3
using StaticArrays
using MimuConstants
using LinearAlgebra
using Printf
using Dates

function crb(w, r, Q_inv)
    r_m = hcat(r...)
    Q_inv_tot = cat(Q_inv...; dims = (1,2))
    Nt =  length(Q_inv)
    Na = length(r)
    Ng = Nt  - Na
    H_m = zeros(3*Nt, 6)
    BcdV3.fill_H_constants!(H_m, Na)
    BcdV3.H!(H_m, r)
    crb = zeros(3, length(w))
    J_m = zeros(3*Nt, 3)

    BcdV3.fill_J_h_buff!(J_m, Na, Ng)
    for n = 1:length(w)
        BcdV3.J_h!(J_m, w[n], r_m, Ng)
        J_tot = hcat(J_m, H_m)
        fim_n = J_tot'*Q_inv_tot*J_tot

        fim_n_inv = inv(fim_n)
        crb[:,n] = diag(fim_n_inv)[1:3]
    end
    return crb
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
r_m = [1.0  0.0 0.0;
-1.0  0.0 0.0;
0.0  1.0 0.0;
0.0 -1.0 0.0]'.*1.0e-2 # [m]
r_m .-= r_m[:,1]
Na = size(r_m, 2)
r = [SVector{3,TT}(r_m[:,k]) for k = 1:Na]

# sig_a = get_rms_noise_acc()
# sig_g = get_rms_noise_gyro()
sig_a = 0.01
sig_g = deg2rad(1.0)

Ng = 4

N = 100 # Dyn
Ns = []
show_every = 100
tol_dlogp = 1.0e-9

Nt = Na + Ng
g_mag = 9.81
Random.seed!(1)

b = [@SVector randn(3) for k = 1:Nt]
T = [I + 1.0e-3*@SMatrix randn(3,3) for k = 1:Nt]
T[1] = I + 1.0e-3*SMatrix{3,3, Float64}(triu(randn(3,3)))

# T = [I + @SMatrix zeros(3,3) for k = 1:Nt]
# b = [@SVector zeros(3) for k = 1:Nt]
# r = [SVector{3}(r_m[:,k])*100 for k = 1:Na]


w_norm = map(x-> 10^x, LinRange(2, 4, N)) .|> deg2rad
# w = [deg2rad(1000.0) * @SVector randn(3) for _ = 1:N]
w_rot = SVector{3}([1.0, 0.0, 0.0])
w = [w_rot*w_norm[n] for n = 1:N]
w_m = hcat(Array.(w)...)
w_dot = [@SVector randn(3) for _ = 1:N]
s = [@SVector randn(3) for _ = 1:N]


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


N_mc = 10^4
dw = zeros(3, N)
dw2 = zeros(3, N)
show_all(x) = show(stdout, "text/plain", x)
Juno.@progress for n = 1:N_mc
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

    w0 = deepcopy(w)
    r_hat, T_hat, b_hat, eta = bcd!(
    copy(r), copy(T), copy(b), y, Q_inv, Ns, g_mag, Val{3*(Na+Ng)}(),Val{Na}(),
    1.0e-12, 100, 1.0e-8, 200, tol_dlogp, 1.0-14, 1; warn = false, w0 = w0
    )
    w_hat_m = hcat(Array.(eta[:w])...)
    dw .+= (w_m  - w_hat_m).^2

    r_hat, T_hat, b_hat, eta = bcd!(
    copy(r), copy(T), copy(b), y2, Q_inv, Ns, g_mag, Val{3*(Na+Ng)}(),Val{Na}(),
    1.0e-12, 100, 1.0e-8, 200, tol_dlogp, 1.0-14, 1; warn = false, w0 = w0.*2
    )
    w_hat_m = hcat(Array.(eta[:w])...)
    dw2 .+= (w_m.*2  - w_hat_m).^2

    # w_hat, w_dot_hat, s_hat  = dynamics(deepcopy(w0), y,  Q_inv, r, Ng,  Val{3*Nt}(), 1.0e-8, 200)
    # w_hat_m2 = hcat(Array.(w_hat)...)
    # dw2 .+= (w_m  - w_hat_m2).^2
end
mse = dw./N_mc
rmse_w_tot = sqrt.(mse)
println("zeta  1")
@show sum(mse)/length(mse)

mse2 = dw2./N_mc
rmse_w_tot2 = sqrt.(mse2)
println("zeta  2")
@show sum(mse2)/length(mse2)
if true
    fig, ax = subplots(1,1)
    crb_w =  sqrt.(crb(w, r, Q_inv))
    for i = 1:3
        ax.plot(w_norm .|> rad2deg, rmse_w_tot[i,:] .|> rad2deg, label = string(i));
        ax.plot(w_norm .|> rad2deg, crb_w[i,:] .|> rad2deg, label = "CRB $(i)");
    end
    ax.plot(w_norm .|> rad2deg, ones(size(w_norm))*sig_g/sqrt(Ng) .|> rad2deg);
    # ax.set_ylim(sig_g |>  rad2deg |> x -> x/10, sig_g |>  rad2deg |> x -> x*10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(true, which = "both")
    ax.legend()
    # ax.set_ylim(0.01, 0.1)
end

nothing
