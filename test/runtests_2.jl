using BcdV3
using StaticArrays
using MimuConstants
using LinearAlgebra
using Printf
using Base.Threads
using Dates

function normalize_diagonals!(T::Vector{MMatrix{3,3,TT,9}}) where {TT}
    D_ii = CartesianIndex.(axes(T[1], 1), axes(T[1], 2))
    N = length(T)
    C = mapreduce(T_n -> tr(T_n), +, T)/(3*N)
    for n = 1:N
        T[n][D_ii] ./= C
    end
    return T
end

function bcd2!(r,T,b, y, Q_inv, Ns, g_mag::TT, ::Val{N_sens},
    ::Val{Na},
    tol_interval, i_max_g, tol_gs, i_max_w, tol_bcd, i_max_bcd;
    callback = nothing) where {TT, N_sens, Na}

    @assert istriu(T[1])
    @assert r[1] == zeros(3)

    N = size(y, 2)
    N_triads = size(y, 1)
    Ng = N_triads - Na
    N_orientions = length(Ns)
    N_dynamics = N - N_orientions

    r_m = zero(MMatrix{3,Na,TT})
    for k = 2:Na
        r_m[:,k] = r[k]
    end
    u = [zero(SVector{3,TT}) for i = 1:N_triads, j = 1:N]
    Qu_inv = [zero(SMatrix{3,3,TT}) for i = 1:N_triads]

    # Statics
    g = [zero(SVector{3,TT}) for _ = 1:N_orientions]

    # Dynamics
    u_m = zero(MVector{N_sens,TT})
    u_n_s = [zero(SVector{N_sens,TT}) for _ = 1:N_dynamics]
    Qu_inv_tall = zero(MMatrix{N_sens, N_sens, TT, N_sens^2})
    # Initial w0 from gyroscopes
    w = map(1:N_dynamics) do n
        mapreduce(k -> y[Na + k, n + N_orientions], +, 1:Ng)/Ng
    end

    # buffs
    w_dot = [zero(SVector{3,TT}) for _ = 1:N_dynamics]
    s = [zero(SVector{3,TT}) for _ = 1:N_dynamics]
    Jrs = [zero(SMatrix{3,3,TT}) for _ = 1:N_dynamics]
    Tg = zero(MMatrix{3,3,TT})

    ii_T_1 = SVector{6,Int}([1,2,3,4,5,6])
    ii_b_1 = SVector{3,Int}([7,8,9])

    ii_T = SVector{9,Int}([1,2,3,4,5,6,7,8,9])
    ii_b = SVector{3,Int}([10,11,12])

    D_ii = CartesianIndex.(axes(T[1], 1), axes(T[1], 2))
    k = 0
    log_p_prev::TT = Inf
    dlogp = zero(TT)
    logp = zero(TT)
    converged = false
    @timeit "Loop" while !converged
        k += 1

        @timeit "u inv" begin
        BcdV3.inverse_affine!(u, y, T, b)
        BcdV3.inverse_affine_Q!(Qu_inv, Q_inv, T)
        end
        @timeit "gravity" begin
        BcdV3.gravity!(g, u, Qu_inv, Na, N_orientions, g_mag, tol_interval, i_max_g)
        end
        @timeit "Qu trans" begin
            for k = 1:N_triads
                kk = 1+3*(k-1):3*k
                Qu_inv_tall[kk,kk] = Qu_inv[k]
            end

        end
        @timeit "u trans" begin
            for n = 1:N_dynamics
                for k = 1:N_triads
                    u_m[1+3(k-1):3*k] = u[k,n+N_orientions]
                end
                u_n_s[n] = SVector(u_m)
            end
        end
        @timeit "dyn" begin

        BcdV3.dynamics!!(w, w_dot, s, u_n_s, SMatrix(Qu_inv_tall), SMatrix(r_m), Ng,
                         tol_gs, i_max_w)

        end

        @timeit "r" begin

            # Positions
            for n = 1:N_dynamics
                Jrs[n] = BcdV3.J_r(w[n], w_dot[n])
            end


            for k = 2:Na
                A_r, a_r = zero(SMatrix{3,3, TT}), zero(SVector{3, TT})
                for n = 1:N_dynamics
                    J = Jrs[n]'*Qu_inv[k]
                    A_r += J*Jrs[n]
                    a_r += J*(u[k,n + N_orientions] - s[n])
                end
                r[k] = A_r\a_r
                r_m[:,k] = r[k]
            end

        end
        @timeit "update u" begin
        # Update u
            @threads for n = 1:N
                for k = 1:N_triads
                    # Static
                    if n <= N_orientions
                        if k <= Na
                            u[k,n] = -g[n]
                        else
                            u[k,n] = zero(SVector{3, TT})
                        end
                    else
                        n_d = n - N_orientions
                        if k <= Na
                            u[k,n] = Jrs[n_d]*r[k] + s[n_d]
                        else
                            u[k,n] = w[n_d]
                        end
                    end
                end
            end
        end
        # Ref T
        @timeit "ref T" begin
            A_Tb_1, a_Tb_1 = zero(SMatrix{9,9, TT}), zero(SVector{9, TT})
            for n = 1:N
                A_1 = BcdV3.J_Tb_ref(u[1,n])
                A_T_Q_inv_1 = A_1'*Q_inv[1]
                if n <= N_orientions
                    A_T_Q_inv_1 *= Ns[n]
                end
                A_Tb_1 += A_T_Q_inv_1*A_1
                a_Tb_1 += A_T_Q_inv_1*y[1,n]
            end
            x_1 = A_Tb_1\a_Tb_1

            T[1] = BcdV3.T_ref(x_1[ii_T_1])
            b[1] = x_1[ii_b_1]
        end
        # T and b estimation
        @timeit "Tb" begin
        for k = 2:size(y,1)
            A_Tb, a_Tb = zero(SMatrix{12,12, TT}), zero(SVector{12, TT})
            for n = 1:size(y,2)
                A = BcdV3.J_Tb(u[k,n])
                A_T_Q_inv = A'*Q_inv[k]
                if n <= N_orientions
                    A_T_Q_inv *= Ns[n]
                end
                A_Tb += A_T_Q_inv*A
                a_Tb += A_T_Q_inv*y[k,n]
            end
            x = A_Tb\a_Tb
            T[k] = SMatrix{3,3, TT}(reshape(x[ii_T], 3,3))
            b[k] = x[ii_b]
        end
        end
        # Normalize Tg
        @timeit "norm" begin
            C::TT = 0.0
            for k = 1:Ng
                C += tr(T[k+Na])
            end
            C /= (3*Ng)
            for k = 1:Ng
                Tg .= MMatrix(T[k+Na])
                for i = 1:3
                    Tg[i,i] /= C
                end
                T[k+Na] = SMatrix(Tg)
            end
        end
        logp = BcdV3.nlogp(y, u, T, b, Q_inv, Ns)
        dlogp = log_p_prev - logp
        log_p_prev = logp

        if callback != nothing
            callback(k, dlogp, logp, r, T, b)
        end

        converged = k == i_max_bcd || abs(dlogp) < tol_bcd
    end
    if k == i_max_bcd
        @warn @sprintf(
            "BCD: Max iterations %d reached. Criterion |dlopp| = %.2e < %.1e = tol \n",
            k, abs(dlogp), tol_bcd
        )
    end
    return r, T, b
end


function get_callback_trace(trace, trace_p, show_every = 1)
    function callback(k::Int, dlogp, log_p, r, T, b)
        if mod(k,show_every) == 0
            time = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
            @printf("%20s %7d logp %.20e  dlogp %.8e\n", time, k, log_p, dlogp)
            # @printf(" C_norm %.8e\n", tr(T[4])/3)
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
N_bcd  = 10^6
show_every = 1000
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


r_hat, T_hat, b_hat = BcdV3.bcd2!(
    r_hat, T_hat, b_hat, y, Q_inv, Ns, g_mag, Val{3*(Na+Ng)}(),Val{Na}(),
    1.0e-12, 100, 1.0e-13, 200, tol_dlogp, 1;
    # callback = callback
    callback = get_callback_trace(trace_log_p, trace_p, show_every) #callback
)

r_hat, T_hat, b_hat = @time BcdV3.bcd2!(
r_hat, T_hat, b_hat, y, Q_inv, Ns, g_mag, Val{3*(Na+Ng)}(),Val{Na}(),
1.0e-12, 100, 1.0e-13, 200, tol_dlogp, N_bcd;
# callback = callback
callback = get_callback_trace(trace_log_p, trace_p, show_every) #callback
)

let
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
