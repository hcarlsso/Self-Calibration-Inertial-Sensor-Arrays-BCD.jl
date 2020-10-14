module BcdV3

using StaticArrays
using LinearAlgebra
using Distributed
using Printf

export
    bcd,
    get_callback,
    get_callback_maxdiff_param,
    Ω


op(x,y) = (x[1] + y[1], x[2] + y[2])

function cat_u(x::Vector{SVector{3,T}}, ::Val{N}) where {T,N}
    copy(reinterpret(SVector{N,T}, x) |> only)
end

function normalize_diagonals!(T::AbstractVector{<:SMatrix{3,3,TT,9}}) where {TT}
    D_ii = CartesianIndex.(axes(T[1], 1), axes(T[1], 2))
    N = length(T)
    C = mapreduce(T_n -> tr(T_n), +, T)/(3*N)
    for n = 1:N
        T_n = zero(MMatrix{3,3,TT,9})
        T_n .= T[n]
        T_n[D_ii] ./= C
        T[n] = SMatrix(T_n)
    end
    return T
end

function suff_stat_Tb_ref(y::Vector{SVector{3,T}}, Q_inv, u, Ns, Tb_out) where {T}
    A_Tb, a_Tb = zero(SMatrix{9,9, T}), zero(SVector{9, T})
    N_orientions = length(Ns)
    for n = 1:length(y)
        A = J_Tb_ref(u[n])
        A_T_Q_inv = A'*Q_inv
        if n <= N_orientions
            A_T_Q_inv *= Ns[n]
        end
        A_Tb += A_T_Q_inv*A
        a_Tb += A_T_Q_inv*y[n]
    end
    put!(Tb_out, (A_Tb, a_Tb))
    nothing
end
function suff_stat_Tb(y::Matrix{SVector{3,T}}, Q_inv, u, Ns, Tb_out) where {T}
    N_orientions = length(Ns)
    for k = 2:size(y,1)
        A_Tb, a_Tb = zero(SMatrix{12,12, T}), zero(SVector{12, T})
        for n = 1:size(y,2)
            A = J_Tb(u[k,n])
            A_T_Q_inv = A'*Q_inv[k]
            if n <= N_orientions
                A_T_Q_inv *= Ns[n]
            end
            A_Tb += A_T_Q_inv*A
            a_Tb += A_T_Q_inv*y[k,n]
        end
        put!(Tb_out[k-1], (A_Tb, a_Tb))
    end
    nothing
end

function nlogp(y::Matrix{SVector{3,type}}, u, T, b, Q_inv, Ns) where {type}
    C = zero(type)
    N = size(y,2)
    N_triads = size(y,1)
    N_orientions = length(Ns)
    for n = 1:N
        for k = 1:N_triads
            e = T[k]*u[k,n] + b[k] - y[k,n]
            c_k_n = dot(e, Q_inv[k], e)
            if n <= N_orientions
                c_k_n *= Ns[n]
            end
            C += c_k_n
        end
    end
    # Average log likelihood
    return C/prod(size(y))
end
function nlogp_naive(y::Matrix{SVector{3,type}}, Q_inv, Ns, r, T, b, g, w, w_dot, s) where {type}
    C = zero(type)
    Na = size(r, 2)
    N = size(y, 2)
    N_triads = size(y,1)
    N_orientions = length(Ns)
    for n = 1:N
        for k = 1:N_triads
            if n <= N_orientions
                if k <= Na
                    u = -g[n]
                else
                    u = zero(SVector{3, type})
                end
            else
                n_d = n - N_orientions
                if k <= Na
                    u = Ω(w[n_d])^2*r[:,k] + Ω(w_dot[n_d])*r[:,k] + s[n_d]
                else
                    u = w[n_d]
                end
            end

            e = T[k]*u + b[k] - y[k,n]
            c_k_n = dot(e, Q_inv[k], e)

            if n <= N_orientions
                c_k_n *= Ns[n]
            end

            C += c_k_n
        end
    end
    return C
end

################################################################################
# Means and jacobians
################################################################################
function Ω(a::SVector{3,T}) where {T}
    A = zero(MMatrix{3,3,T, 9})

    A[1,2] = -a[3]
    A[2,1] =  a[3]

    A[1,3] =  a[2]
    A[3,1] = -a[2]

    A[2,3] = -a[1]
    A[3,2] =  a[1]

    return SMatrix(A)
end
J_r(w, w_dot) = Ω(w)^2 + Ω(w_dot)

function J_Tb(u::SVector{3,T}) where {T}
    A = zero(MMatrix{3,12,T})

    for i = 1:3, j = 1:3
        A[j,j + 3(i-1)] = u[i]
    end
    for i = 1:3
        A[i,i+9] = one(T)
    end
    SMatrix(A)
end
function J_Tb_ref(u::SVector{3,T}) where {T}
    A = zero(MMatrix{3,9,T})
    A[1,1] = u[1]

    A[1,2] = u[2]
    A[2,3] = u[2]

    A[1,4] = u[3]
    A[2,5] = u[3]
    A[3,6] = u[3]

    A[1,7] = one(T)
    A[2,8] = one(T)
    A[3,9] = one(T)
    SMatrix(A)
end
function T_ref(x::SVector{6,type}) where {type}
    T = zero(MMatrix{3,3,type})
    T[1,1] = x[1]
    T[1,2] = x[2]
    T[1,3] = x[4]

    T[2,2] = x[3]
    T[2,3] = x[5]

    T[3,3] = x[6]
    SMatrix(T)
end
function H!(H::AbstractMatrix{T}, r) where {T}
    Na = size(r, 2)
    for k = 1:Na
        H[1+3(k-1):3*k,1:3] = -Ω(r[:,k])
    end
    nothing
end
function fill_H_constants!(H::AbstractMatrix{T}, Na) where {T}
    eye = SMatrix{3,3, T}(I)
    for k = 1:Na
        H[1+3(k-1):3*k, 4:6] = eye
    end
    nothing
end
function h!(y::AbstractVector, w::AbstractVector, r::AbstractMatrix, Ng::Int)

    Na = size(r, 2)
    @inbounds for k = 1:Na
        o = 3(k-1)
        y[1+o] = -w[2]^2*r[1,k]
        y[1+o] -= w[3]^2*r[1,k]
        y[1+o] += w[1]*w[2]*r[2,k]
        y[1+o] += w[1]*w[3]*r[3,k]

        y[2+o] = w[1]*w[2]*r[1,k]
        y[2+o] -= w[1]^2*r[2,k]
        y[2+o] -= w[3]^2*r[2,k]
        y[2+o] += w[2]*w[3]*r[3,k]

        y[3+o] = w[1]*w[3]*r[1,k]
        y[3+o] += w[2]*w[3]*r[2,k]
        y[3+o] -= w[1]^2*r[3,k]
        y[3+o] -= w[2]^2*r[3,k]
    end
    @inbounds for k = 1:Ng
        o = 3(k-1) + 3*Na
        y[1+o] = w[1]
        y[2+o] = w[2]
        y[3+o] = w[3]
    end
    nothing
end
function J_h!(J::AbstractMatrix{T}, w, r, Ng) where {T}
    # fill!(J, zero(T))
    Na = size(r, 2)
    @inbounds for i = 1:3
        @inbounds for k = 1:Na
            o = 3(k-1)
            if i == 1
                J[1+o, i]  = w[3]*r[3,k] +   w[2]*r[2,k]
                J[2+o, i]  = w[2]*r[1,k] - 2.0*w[1]*r[2,k]
                J[3+o, i]  = w[3]*r[1,k] - 2.0*w[1]*r[3,k]
            elseif i == 2
                J[1+o, i]  = w[1]*r[2,k] - 2.0*w[2]*r[1,k]
                J[2+o, i]  = w[3]*r[3,k] +   w[1]*r[1,k]
                J[3+o, i]  = w[3]*r[2,k] - 2.0*w[2]*r[3,k]
            else
                J[1+o, i]  = w[1]*r[3,k] - 2.0*w[3]*r[1,k]
                J[2+o, i]  = w[2]*r[3,k] - 2.0*w[3]*r[2,k]
                J[3+o, i]  = w[2]*r[2,k] +   w[1]*r[1,k]
            end
        end
    end
    nothing
end
function fill_J_h_buff!(J::AbstractMatrix{T}, Na, Ng) where {T}
    @inbounds for i = 1:3
        @inbounds for k = i:3:3*Ng
            J[3*Na + k,i] = one(T)
        end
    end
    nothing
end
################################################################################
# Data manipulation
################################################################################
function inverse_affine!(u, y, T, b)
    for n = 1:size(y,2)
        for k = 1:size(y,1)
            u[k,n] = T[k]\(y[k,n] - b[k])
        end
    end
    nothing
end
function inverse_affine_Q!(Qu_inv, Q_inv, T)
    for n = 1:length(Q_inv)
        Qu_inv[n] = T[n]'*Q_inv[n]*T[n]
    end
    nothing
end

################################################################################
# Gravity estimator
################################################################################
"""
    Should multiply D and d with number of measurements, but they take out each other.
"""
function gravity!(g::Vector{SVector{3,T}}, u, Qu_inv, Na, Nt, g_mag, tol, i_max) where {T}
    inds = @SVector [1,3]

    D::SMatrix{3,3,T} = mapreduce(k -> Qu_inv[k], +, 1:Na)
    (sig_max, sig_min) = svdvals(D)[inds]
    for n = 1:Nt
        d::SVector{3, T} = mapreduce(k -> Qu_inv[k]*u[k,n], +, 1:Na)
        g[n] = estimate_gravity(d, D, sig_min, sig_max, g_mag, tol, i_max)
    end
    nothing
end
function estimate_gravity(d::SVector{3,T}, D::SMatrix{3,3,T}, s_min, s_max, g_mag,
    tol, iter_max) where {T}

    g_mid = zeros(SVector{3,T})
    d_norm = norm(d)
    lambda_up = d_norm/g_mag - s_min
    g_low = norm((D + lambda_up*I)\d)

    lambda_low = d_norm/g_mag - s_max
    g_upper = norm((D + lambda_low*I)\d)

    k = 0
    converged = false
    while !converged
        k += 1
        lambda_mid = (lambda_up + lambda_low)/2.0
        g_mid = -(D + lambda_mid*I)\d
        g_mid_norm = norm(g_mid)

        if g_mid_norm > g_mag
            (lambda_low, g_upper) = (lambda_mid, g_mid_norm)
        else
            (lambda_up, g_low) = (lambda_mid, g_mid_norm)
        end
        # lower can be higher than upper, why?
        converged = k == iter_max || abs(g_upper - g_low) < tol

        # @printf(
        # "Bisection: k = %3d |g_up - g_low| = %.5e < %.1e = tol \n", k, abs(g_upper - g_low), tol
        # )
    end
    if k == iter_max
        @warn @sprintf(
            "Bisection: Max iterations %d reached. Criterion |g_up - g_low| = %.2e < %.1e = tol \n",
            k, abs(g_upper - g_low), tol
        )
    end
    return g_mid::SVector{3,T}
end
################################################################################
# Gauss Newnton
################################################################################
"""
    Need to put e, y_pred and J_m in arguments so that the type is concrete.
    """
function gauss_newton(w::SVector{3,T}, e::SVector{N,T}, y, P, r, Ng::Int, g_tol, i_max,
    y_pred, J_m) where {N, T}


    H = zero(SMatrix{3, 3, T})
    g = zero(SVector{3, T})
    p = zero(SVector{3, T})

    k = 0::Int
    converged = false

    while !converged
        k += 1

        h!(y_pred, w, r, Ng)
        e = y - SVector(y_pred)

        J_h!(J_m, w, r, Ng)
        J = SMatrix(J_m)


        J_T_P = J'*P
        H = J_T_P*J
        g = J_T_P*e

        p = H\g
        w += p

        # ||Jp||/(1+||r||
        # Should acutally be  ||RJp||/(1+||Rr||  where R'R = P
        converged = k == i_max || norm(J*p, Inf) <  g_tol*(1.0 + norm(e, Inf))

    end
    if k == i_max
        crit = norm(SMatrix(J_m)*p, Inf)/(1.0 + norm(e, Inf))

        @warn @sprintf(
            "GS: Max iterations %d reached. Criterion ||Jp||/(1+||r||) = %.2e < %.1e = tol \n",
            k, crit, g_tol
        )
    end


    w::SVector{3,T}
end
function dynamics!!(w::Vector{SVector{3,T}}, w_dot::Vector{SVector{3,T}}, s::Vector{SVector{3,T}},
    u::Vector{SVector{N,T}}, Qu_inv::SMatrix{N,N,T},
    r, Ng, g_tol, i_max) where{N,T}
    N_dynamics = length(w)

    H_m = zero(MMatrix{N, 6, T})
    Na = size(r, 2)
    fill_H_constants!(H_m, Na)
    H!(H_m, r)
    H = SMatrix(H_m)
    e = zero(SVector{N, T})

    phi = zero(SVector{6, T})
    inds_w_dot = @SVector [1,2,3]
    inds_s = @SVector [4,5,6]

    # Buffs
    y_pred = zero(MVector{N, T})
    J_m = zero(MMatrix{N, 3, T})
    fill_J_h_buff!(J_m, Na, Ng)

    H_T_Q_inv = H'*Qu_inv
    WLS = (H_T_Q_inv*H)\H_T_Q_inv
    P = Qu_inv - H_T_Q_inv'*WLS


    for n = 1:N_dynamics
        w[n] = gauss_newton(w[n], e, u[n], P, r, Ng, g_tol, i_max, y_pred, J_m)
        h!(y_pred, w[n], r, Ng)
        e = u[n] - SVector(y_pred)
        phi = WLS*e
        w_dot[n] = phi[inds_w_dot]
        s[n] = phi[inds_s]
    end
    w, w_dot, s
end

################################################################################
# BCD loop
################################################################################
function do_work(y::Matrix{SVector{3, TT}}, Q_inv, Ns, ::Val{N_sens}, ::Val{Na},
    Ng, g_mag,
    job, r_in, r_out, Tb_in, Tb_out, nlogp_ch,
    tol_interval, i_max_g, tol_gs, i_max_w) where {TT, N_sens, Na}


    N = size(y, 2)
    N_triads = Na + Ng
    N_orientions = length(Ns)
    N_dynamics = N - N_orientions

    println("Got # dynamic samples: ", N_dynamics)
    println("Got # orientations: ", N_orientions)

    u = [zero(SVector{3,TT}) for i = 1:N_triads, j = 1:N]
    Qu_inv = [zero(SMatrix{3,3,TT}) for i = 1:N_triads]

    # Statics
    g = [zero(SVector{3,TT}) for _ = 1:N_orientions]

    # Dynamics
    u_n = [zero(SVector{N_sens,TT}) for _ = 1:N_dynamics]
    # Qu_inv_tall = zero(SMatrix{N_sens, N_sens, TT})
    # Initial w0 from gyroscopes
    w = map(1:N_dynamics) do n
        mapreduce(k -> y[Na + k, n + N_orientions], +, 1:Ng)/Ng
    end
    w_dot = [zero(SVector{3,TT}) for _ = 1:N_dynamics]
    s = [zero(SVector{3,TT}) for _ = 1:N_dynamics]
    Jrs = [zero(SMatrix{3,3,TT}) for _ = 1:N_dynamics]

    # Initial values
    r::SMatrix{3,Na,TT} = take!(r_in)
    T::Vector{SMatrix{3,3,TT}}, b::Vector{SVector{3,TT}} = take!(Tb_in)

    println("Start worker")
    println("r0")
    display(r)
    println("T")
    foreach(t -> display(t), T)
    println("b")
    foreach(t -> display(t), b)
    # @printf("%20s %.8e\n", "Start ", nlogp_naive(y, Q_inv, Ns, r, T, b, g, w, phi))
    done = false
    while !done

        job_type = take!(job) # Blocking call
        if job_type == :eta_r
            inverse_affine!(u, y, T, b)
            inverse_affine_Q!(Qu_inv, Q_inv, T)

            # @printf("%20s %.16e\n", "u", nlogp_naive(
            #     y, Q_inv, Ns, r, T, b, g, w, w_dot, s
            # ))
            if N_orientions > 0
                gravity!(g, u, Qu_inv, Na, N_orientions, g_mag, tol_interval, i_max_g)
            end
            # @printf("%20s %.16e\n", "g", nlogp_naive(
            #     y, Q_inv, Ns, r, T, b, g, w, w_dot, s
            # ))


            # Dynamics
            Qu_inv_tall::SMatrix{N_sens, N_sens, TT, N_sens^2} =
                SMatrix{N_sens, N_sens, TT, N_sens^2}(cat(Qu_inv...; dims = (1,2)))
            for n = 1:N_dynamics
                u_n[n] = cat_u(u[:,n+N_orientions], Val{N_sens}())
            end

            dynamics!!(w, w_dot, s, u_n, Qu_inv_tall, r, Ng, tol_gs, i_max_w)

            # @printf("%20s %.16e\n", "Dynamics", nlogp_naive(y, Q_inv, Ns, r, T, b, g, w, w_dot, s))

            # Positions
            for n = 1:N_dynamics
                Jrs[n] = J_r(w[n], w_dot[n])
            end
            for k = 2:Na
                A_r, a_r = zero(SMatrix{3,3, TT}), zero(SVector{3, TT})
                for n = 1:N_dynamics
                    J = Jrs[n]'*Qu_inv[k]
                    A_r += J*Jrs[n]
                    a_r += J*(u[k,n + N_orientions] - s[n])
                end
                put!(r_out[k-1], (A_r, a_r))
            end
        elseif job_type == :Tb
            r = take!(r_in)::SMatrix{3,Na,TT}
            # println("Update r worker:")
            # display(r)
            # @printf("%20s %.16e\n", "Pos", nlogp_naive(y, Q_inv, Ns, r, T, b, g, w, w_dot, s))
            # Update u
            for n = 1:N, k = 1:N_triads
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
                        u[k,n] = Jrs[n_d]*r[:,k] + s[n_d]
                    else
                        u[k,n] = w[n_d]
                    end
                end
            end
            suff_stat_Tb_ref(y[1,:], Q_inv[1], u[1,:], Ns, Tb_out[1])
            suff_stat_Tb(y, Q_inv, u, Ns, Tb_out[2:end])
        elseif job_type == :nlopg
            T, b = take!(Tb_in)
            # take!(Tb_in)
            # @printf("%20s %.16e\n", "Tb", nlogp_naive(y, Q_inv, Ns, r, T, b, g, w, w_dot, s))
            C = nlogp(y, u, T, b, Q_inv, Ns)
            put!(nlogp_ch, C)
        elseif job_type == :done
            done = true
            println("Shutdown")
        else
            throw(ErrorException("Incorrect phase: $(job)"))
        end
    end
    nothing
end
function bcd(r0, T0, b0, y_s, y_d, Q_inv, Ns, Na , Ng, g_mag,
    tol_bcd, i_max_bcd, tol_interval, i_max_g, tol_gs, i_max_w,
    pool = workers(); callback = nothing)

    Nt = Na + Ng

    @assert istriu(T0[1])
    @assert r0[:,1] == zeros(3)


    TT = Float64
    T_3x3 = SMatrix{3,3,TT}
    V_3 = SVector{3, TT}

    remote(x) = RemoteChannel(()-> x)

    job = [Channel{Symbol}(4) |> remote for p in pool]
    r_in = [Channel{typeof(r0)}(1) |> remote for p in pool]
    r_out = [Channel{Tuple{T_3x3, V_3}}(1) |> remote for k = 1:Na-1, p in pool]
    Tb_in = [Channel{Tuple{Vector{T_3x3}, Vector{V_3}}}(1) |> remote for p in pool]
    Tb_out = map(pool) do p
        vcat(
        Channel{Tuple{SMatrix{9,9,TT}, SVector{9, TT}}}(1) |> remote,
        [Channel{Tuple{SMatrix{12,12,TT}, SVector{12, TT}}}(1) |> remote for k = 1:Nt - 1]
        )
    end |> x -> hcat(x...)
    logp_out = [Channel{TT}(1) |> remote for p in pool]

    N_w = length(pool)
    # Initial values
    T = deepcopy(T0)
    b = deepcopy(b0)
    r = deepcopy(r0)
    @async for n in 1:N_w
        put!(r_in[n], r)
        put!(Tb_in[n], (T, b))
    end

    (size_s, rest_s) = fldmod(size(y_s,2), N_w)
    (size_d, rest_d) = fldmod(size(y_d,2), N_w)

    for (k,p) in enumerate(pool)
        if k == N_w
            y_p = hcat(y_s[:,1 + size_s*(k-1):end], y_d[:,1 + size_d*(k-1):end])
            Ns_p = Ns[1+size_s*(k-1):end]
        else
            slice_s = range(1 + size_s*(k-1), length = size_s)
            slice_d = range(1 + size_d*(k-1), length = size_d)
            y_p = hcat(y_s[:,slice_s], y_d[:,slice_d])
            Ns_p = Ns[slice_s]
        end
        remote_do(do_work, p, y_p, Q_inv, Ns_p, Val{3*(Na+Ng)}(), Val{Na}(), Ng,
                  g_mag, job[k], r_in[k], r_out[:,k], Tb_in[k], Tb_out[:,k], logp_out[k],
                  tol_interval, i_max_g, tol_gs, i_max_w)
    end

    r_list = [@SVector zeros(TT,3) for k=1:Na]

    k = 0
    log_p_prev::TT = Inf
    converged = false
    while !converged
        k += 1
        @sync foreach(j -> put!(j, :eta_r), job)

        # Workers do eta and r
        for k = 1:Na-1
            A_r, a_r = @sync mapreduce(n -> take!(r_out[k,n]), op, 1:N_w)
            r_list[k+1] = A_r\a_r
        end
        r = hcat(r_list...)
        # display(r)
        @sync foreach(p -> put!(p, r), r_in)


        @sync foreach(j -> put!(j, :Tb), job)
        for k = 1:Nt
            A_Tb, a_Tb = @sync mapreduce(n -> take!(Tb_out[k,n]), op, 1:N_w)
            x = A_Tb\a_Tb
            if k == 1
                T[k] = T_ref(x[SVector{6,Int}(1:6 |> collect)])
                b[k] = x[SVector{3,Int}(7:9 |> collect)]
            else
                T[k] = SMatrix{3,3}(reshape(x[1:9], 3,3))
                b[k] = x[SVector{3,Int}(10:12 |> collect)]
            end
        end
        @views normalize_diagonals!(T[1+Na:end])
        foreach(p -> put!(p, (T,b)), Tb_in)


        foreach(j -> put!(j, :nlopg), job)
        log_p_val = @sync mapreduce(n-> take!(logp_out[n]), +, 1:N_w)
        dlogp = log_p_prev - log_p_val

        log_p_prev = log_p_val

        if callback != nothing
            callback(k, dlogp, log_p_val, r, T, b)
        end

        converged = k == i_max_bcd || abs(dlogp) < tol_bcd
    end
    foreach(j -> put!(j, :done), job)
    return r, T, b
end

function get_callback(show_every = 1)
    function callback(k::Int, dlogp, log_p, r, T, b)
        if mod(k,show_every) == 0
            @printf("%15d logp %.20e  dlogp %.8e\n", k, log_p, dlogp)
            # @printf(" C_norm %.8e\n", tr(T[4])/3)
        end
    end
    return
end
function get_callback_maxdiff_param(r_true, T_true, b_true, show_every = 1)

    callback(k::Int, dlogp, log_p, r, T, b) = begin
        if mod(k, show_every) == 0
            r_e = norm(r - r_true, Inf)
            T_e = [norm(T[k] - T_true[k]) for k = 1:length(T)] |> x  -> norm(x, Inf)
            b_e = [norm(b[k] - b_true[k]) for k = 1:length(b)] |> x  -> norm(x, Inf)
            @printf("%15d logp %.3e  dlogp %.3e r_e %.4e T_e %.4e b_e %.4e \n", k, log_p, dlogp,
            r_e, T_e, b_e)
            # @printf(" C_norm %.8e\n", tr(T[4])/3)
        end
        # println(k, " ", log_p)
    end

    return callback

end

end # module
