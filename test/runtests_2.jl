using BcdV3
using StaticArrays
using MimuConstants
using LinearAlgebra
using Printf
using Base.Threads
using Dates
using TimerOutputs

f_relchange(f_x::T, f_x_previous) where T = abs(f_x - f_x_previous)/abs(f_x)
function maxdiff(x::AbstractArray, y::AbstractArray)
    res = real(zero(x[1] - y[1]))
    @inbounds for i in 1:length(x)
        delta = abs(x[i] - y[i])
        if delta > res
            res = delta
        end
    end
    return res
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
    N_bcd  = 2*10^6
    show_every = 1000
    tol_dlogp = 0.0
    tol_dx = 1.0e-14

    # pool = workers()
    Nt = Na + Ng
    g_mag = 9.81


    g = [SVector{3,TT}(g_m[:,k]) for k = 1:N_orientions] .|> x -> x/norm(x)*g_mag

    b = [@SVector randn(3) for k = 1:Nt]
    T = [I + 1.0e-3*@SMatrix randn(3,3) for k = 1:Nt]
    T[1] = I + 1.0e-3*SMatrix{3,3, TT}(triu(randn(3,3)))

    Cs_prior = map(1:Ng) do k
        tr(T[k + Na])
    end
    @printf("Prior C_norm %.8e\n", sum(Cs_prior)/(3*Ng))

    local C = 0.0
    for k = 1:Ng
        C += tr(T[k+Na])
    end
    C /= (3*Ng)
    for k = 1:Ng
        Tg = MMatrix(T[k+Na])
        for i = 1:3
            Tg[i,i] /= C
        end
        T[k+Na] = SMatrix(Tg)
    end
    Cs = map(1:Ng) do k
        tr(T[k + Na])
    end
    @printf("Post  C_norm %.8e\n", sum(Cs)/(3*Ng))

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

    trace = Dict(
        :f => zeros(N_bcd),
        :f_abs => zeros(N_bcd),
        :f_rel => zeros(N_bcd),
        :x_abs => zeros(N_bcd)
    )
    trace_p = Dict(:r => [], :T => [], :b => [], :k => [])

    println("BCD v3")


    bcd!(
        r_hat, T_hat, b_hat, y, Q_inv, Ns, g_mag, Val{3*(Na+Ng)}(),Val{Na}(),
        1.0e-12, 100, 1.0e-13, 200, tol_dlogp, tol_dx, 1
    )

    # to = TimerOutput()
    r_hat, T_hat, b_hat = @time  bcd!(
        r_hat, T_hat, b_hat, y, Q_inv, Ns, g_mag, Val{3*(Na+Ng)}(),Val{Na}(),
        1.0e-12, 100, 1.0e-13, 200, tol_dlogp, tol_dx, N_bcd;
        callback = get_callback_trace(
            trace, trace_p, show_every
        ) #callback
    )

    nothing
end

if true
    fig, ax = subplots(1,1)
    ax.plot(trace[:f])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(true)
    ax.set_title("logp")
end

if true

    fig, ax = subplots(1,1)
    ax.plot(trace[:f_rel])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(true)
    ax.set_title("logp rel")
end
if true
    fig, ax = subplots(1,1)
    data = trace[:f_abs]
    mask = abs.(data) .< 1.0e-16
    data[mask] .= 1.0e-16
    # ax.plot(trace[:f_abs][end-200_000:end])
    ax.plot(data[1:10:end])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(true)
    ax.set_title("-diff logp")
end
if true
    fig, ax = subplots(1,1)
    N_window = 1000
    data = zeros(length(trace[:f_abs])-N_window)
    for k = 1:length(trace[:f_abs])-N_window
        data[k] = maximum(trace[:f_abs][k:k + N_window])
    end
    # ax.plot(trace[:f_abs][end-200_000:end])
    ax.plot(data)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(true)
    ax.set_title("-diff logp windows")
end
if true
    fig, ax = subplots(1,1)
    ax.plot(trace[:x_abs])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(true)
    ax.set_title("Norm(dx)")
end

if true
    indices = trace_p[:k]

    fig, ax = subplots(1,1)
    for k = 4:4
        r_trace = cat([Array(r_t[k] - r[k])  for r_t in trace_p[:r]]...; dims = 2)
        for i = 1:3
            ax.plot(indices, abs.(r_trace[i,:]), label = "$(i)")
        end
    end
    ax.grid(true)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
end
if true
    indices = trace_p[:k]

    fig, ax = subplots(1,1)
    k = 9
    T_trace = cat([Array(T_t[k] - T[k])[:]  for T_t in trace_p[:T]]...; dims = 2)
    for i = 1:9
        ax.plot(indices, abs.(T_trace[i,:]), label = "$(i)")
    end
    ax.grid(true)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
end
