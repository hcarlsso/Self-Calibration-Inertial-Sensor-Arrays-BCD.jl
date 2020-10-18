using BcdV3
using StaticArrays
using MimuConstants
using LinearAlgebra
using Printf
using Base.Threads
using Dates
using TimerOutputs

function get_callback_trace(trace, trace_p, show_every = 1)
    function callback(k::Int, dlogp, log_p, r, T, b)
        if mod(k,show_every) == 0
            time = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
            @printf("%20s %7d logp %.20e  dlogp %.8e\n", time, k, log_p, dlogp)

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
        trace[k] = log_p
    end
    return callback
end

# test_func() = begin
if true
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

    N = 50 # Dyn
    N_per_orientation = Int(N/5)
    Ns = ones(Int, N_orientions).*N_per_orientation
    N_bcd  = 10^3
    show_every = 100
    tol_dlogp = 1.0e-14

    # pool = workers()
    Nt = Na + Ng
    g_mag = 9.81


    g = [SVector{3,TT}(g_m[:,k]) for k = 1:N_orientions] .|> x -> x/norm(x)*g_mag

    b = [@SVector randn(3) for k = 1:Nt]
    T = [I + 1.0e-3*@SMatrix randn(3,3) for k = 1:Nt]
    T[1] = I + 1.0e-3*SMatrix{3,3, TT}(triu(randn(3,3)))


    r = [SVector{3,TT}(r_m[:,k]) for k = 1:Na]

    w = [deg2rad(1000.0) * @SVector randn(3) for _ = 1:N]
    w_dot = [deg2rad(40e3) * @SVector randn(3) for _ = 1:N]
    s = [20.0 * @SVector randn(3) for _ = 1:N]

    y_s_true = vcat(
        [T[k]*(-g[n]) + b[k] for k = 1:Na, n = 1:N_orientions],
        [b[k] for k = range(Na+1,length=Ng), n = 1:N_orientions],
    )
    y_d_true = vcat(
        [T[k]*(Ω(w[n])^2*r[k] + Ω(w_dot[n])*r[k] + s[n]) + b[k] for k = 1:Na, n = 1:N],
        [T[k]*w[n]+b[k] for k = range(Na+1,length=Ng), n = 1:N]
    )

    y_s = [y_s_true[k,n] + 0.0*(k <= Na ? sig_a : sig_g)*randn(3)/sqrt(N_per_orientation) for k = 1:Nt, n = 1:N_orientions]
    y_d = [y_d_true[k,n] + 0.0*( k <= Na ? sig_a : sig_g)*randn(3) for k = 1:Nt, n = 1:N]

    y = hcat(y_s, y_d)
    Q_inv = [SMatrix{3,3,TT}(I)/( k <= Na ? sig_a^2 :  sig_g^2) for k = 1:Nt]

    b0 = [b_n + 1.0e-2*randn(3) for  b_n in b]
    T0 = [T[k] + 1.0e-3*(k == 1 ? triu(randn(3,3)) : randn(3,3))  for k = 1:Nt]
    r0 = deepcopy(r)
    for k = 2:Na
        r0[k] = r0[k] + 1.0e-3*randn(3)
    end

    r_hat = deepcopy(r0)
    T_hat = deepcopy(T0)
    b_hat = deepcopy(b0)

    trace_log_p = zeros(N_bcd)
    trace_p = Dict(:r => [], :T => [], :b => [], :k => [])

    println("BCD v2")


    r_hat, T_hat, b_hat = bcd!(
        r_hat, T_hat, b_hat, y, Q_inv, Ns, g_mag, Val{3*(Na+Ng)}(),Val{Na}(),
        1.0e-12, 100, 1.0e-13, 200, tol_dlogp, 1;
        # callback = callback
        callback = get_callback_trace(trace_log_p, trace_p, show_every) #callback
    )

    to = TimerOutput()
    r_hat, T_hat, b_hat = @time  bcd!(
        r_hat, T_hat, b_hat, y, Q_inv, Ns, g_mag, Val{3*(Na+Ng)}(),Val{Na}(),
        1.0e-12, 100, 1.0e-13, 200, tol_dlogp, N_bcd;
        # callback = callback
        callback = get_callback_trace(trace_log_p, trace_p, show_every) #callback
    )
    show(to)

    if false
        fig, ax = subplots(1,1)
        ax.plot(-diff(trace_log_p[trace_log_p .> 0]))
        ax.set_xscale("log")
        ax.set_yscale("log")
    end
    #
    # let
    #     N_sens = Val{3*(Na+Ng)}()
    #     Val_Na = Val{Na}()
    #     @code_warntype bcd2!(
    #     r_hat, T_hat, b_hat, y, Q_inv, Ns, g_mag, N_sens, Val_Na,
    #     1.0e-12, 100, 1.0e-13, 200, tol_dlogp, N_bcd;
    #     # callback = callback
    #     callback = get_callback_trace(trace_log_p, trace_p, show_every) #callback
    # )
    # end
    nothing

end

# @code_warntype test_func()
