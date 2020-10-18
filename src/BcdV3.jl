module BcdV3

using StaticArrays
using LinearAlgebra
using Distributed
using Printf
using Base.Threads

export
    bcd!,
    get_callback,
    get_callback_maxdiff_param,
    Ω,
    inverse_affine_Q,
    inverse_affine,
    dynamics


function nlogp(y::Matrix{SVector{3,type}}, u, T, b, Q_inv, Ns) where {type}
    C = Atomic{type}(0.0)
    N = size(y,2)
    N_triads = size(y,1)
    N_orientions = length(Ns)
    @threads for n = 1:N
        for k = 1:N_triads
            e = T[k]*u[k,n] + b[k] - y[k,n]
            c_k_n = dot(e, Q_inv[k], e)
            if n <= N_orientions
                c_k_n *= Ns[n]
            end
            # C += c_k_n
            atomic_add!(C,c_k_n)
        end
    end
    # Average log likelihood
    return C[]/prod(size(y))
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
function H!(H::AbstractMatrix{T}, r::Vector{SVector{3,T}}) where {T}
    Na = length(r)
    for k = 1:Na
        H[1+3(k-1):3*k,1:3] = -Ω(r[k])
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
function inverse_affine(y, T, b)
    u = [similar_type(y[k,n]) for k = 1:size(y,1), n = 1:size(y,2)]
    inverse_affine!(u, y, T, b)
    return u
end
function inverse_affine!(u, y, T, b)
    @threads for n = 1:size(y,2)
        for k = 1:size(y,1)
            u[k,n] = T[k]\(y[k,n] - b[k])
        end
    end
    nothing
end
function inverse_affine_Q(Q_inv, T)
    Qu_inv = [similar_type(Q_k) for Q_k in Q_inv]
    inverse_affine_Q!(Qu_inv, Q_inv, T)
    return Qu_inv
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
function dynamics(w0::Vector{SVector{3,T}}, u, Qu_inv, r, Ng, ::Val{N_sens},
    tol_gs, i_max_w) where {T, N_sens}

    N_dynamics = length(w0)

    w_dot = [zero(SVector{3,T}) for _ = 1:N_dynamics]
    s = [zero(SVector{3,T}) for _ = 1:N_dynamics]

    Qu_inv_tall::SMatrix{N_sens, N_sens, T, N_sens^2} =
        SMatrix{N_sens, N_sens, T, N_sens^2}(cat(Qu_inv...; dims = (1,2)))
    u_n = [zero(SVector{N_sens,T}) for _ = 1:N_dynamics]
    for n = 1:N_dynamics
        u_n[n] = cat_u(u[:,n], Val{N_sens}())
    end
    w = [copy(w0_n) for w0_n in w0]
    dynamics!!(w, w_dot, s, u_n, Qu_inv_tall, r, Ng, tol_gs, i_max_w)

    return w, w_dot, s
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
# Callbacks
################################################################################
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
################################################################################
# BCD loop
################################################################################
function bcd!(r,T,b, y, Q_inv, Ns, g_mag::TT, ::Val{N_sens},
    ::Val{Na},
    tol_interval, i_max_g, tol_gs, i_max_w, tol_bcd, i_max_bcd;
    callback = nothing) where {TT, N_sens, Na}

    @assert istriu(T[1])
    @assert r[1] == zeros(3)
    @assert length(T) == length(b) == size(y,1) == length(Q_inv)
    @assert length(r) == Na

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
    u_n = [zero(SVector{N_sens,TT}) for _ = 1:N_dynamics]
    Qu_inv_tall = zero(MMatrix{N_sens, N_sens, TT, N_sens^2})
    # Initial w0 from gyroscopes
    w = map(1:N_dynamics) do n
        mapreduce(k -> y[Na + k, n + N_orientions], +, 1:Ng)/Ng
    end

    w_dot = [zero(SVector{3,TT}) for _ = 1:N_dynamics]
    s = [zero(SVector{3,TT}) for _ = 1:N_dynamics]

    # WLS = zero(SMatrix{6, N_sens, TT})
    H_m = zero(MMatrix{N_sens, 6, TT})
    fill_H_constants!(H_m, Na)

    # Buffs
    y_pred = [zero(MVector{N_sens, TT}) for _ = 1:nthreads()]
    J_m = map(1:nthreads()) do n
        J = zero(MMatrix{N_sens, 3, TT})
        fill_J_h_buff!(J, Na, Ng)
        J
    end

    Jrs = [zero(SMatrix{3,3,TT}) for _ = 1:N_dynamics]
    Tg = zero(MMatrix{3,3,TT})

    ii_T_1 = SVector{6,Int}([1,2,3,4,5,6])
    ii_b_1 = SVector{3,Int}([7,8,9])

    ii_T = SVector{9,Int}([1,2,3,4,5,6,7,8,9])
    ii_b = SVector{3,Int}([10,11,12])

    inds_w_dot = SVector{3,Int}([1,2,3])
    inds_s = SVector{3,Int}([4,5,6])

    k = 0
    log_p_prev::TT = Inf
    dlogp = zero(TT)
    logp = zero(TT)
    converged = false
    while !converged
        k += 1

        inverse_affine!(u, y, T, b)
        inverse_affine_Q!(Qu_inv, Q_inv, T)

        gravity!(g, u, Qu_inv, Na, N_orientions, g_mag, tol_interval, i_max_g)

        for k = 1:N_triads
            kk = 1+3*(k-1):3*k
            Qu_inv_tall[kk,kk] = Qu_inv[k]
        end
        for n = 1:N_dynamics
            for k = 1:N_triads
                u_m[1+3(k-1):3*k] = u[k,n+N_orientions]
            end
            u_n[n] = SVector(u_m)
        end


        H!(H_m, r)
        H = SMatrix(H_m)
        H_T_Q_inv = H'*SMatrix(Qu_inv_tall)
        H_T_Q_inv_H = H_T_Q_inv*H
        WLS = H_T_Q_inv_H\H_T_Q_inv
        P2 = H_T_Q_inv'*WLS
        P = SMatrix(Qu_inv_tall) - P2

        @threads for n = 1:N_dynamics
            e = zero(SVector{N_sens, TT})
            phi = zero(SVector{6, TT})
            y_pred_i = y_pred[threadid()]
            J_m_i = J_m[threadid()]

            w[n] = gauss_newton(w[n], e, u_n[n], P, r_m, Ng, tol_gs, i_max_w,
            y_pred_i, J_m_i)

            h!(y_pred_i, w[n], r_m, Ng)
            e = u_n[n] - SVector(y_pred_i)
            phi = WLS*e
            w_dot[n] = phi[inds_w_dot]
            s[n] = phi[inds_s]
        end

        # Positions
        @threads for n = 1:N_dynamics
            Jrs[n] = J_r(w[n], w_dot[n])
        end
        @threads for k = 2:Na
            A_r, a_r = zero(SMatrix{3,3, TT}), zero(SVector{3, TT})
            for n = 1:N_dynamics
                J = Jrs[n]'*Qu_inv[k]
                A_r += J*Jrs[n]
                a_r += J*(u[k,n + N_orientions] - s[n])
            end
            r[k] = A_r\a_r
            r_m[:,k] = r[k]
        end

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

        # Ref Tb
        A_Tb_1, a_Tb_1 = zero(SMatrix{9,9, TT}), zero(SVector{9, TT})
        for n = 1:N
            A_1 = J_Tb_ref(u[1,n])
            A_T_Q_inv_1 = A_1'*Q_inv[1]
            if n <= N_orientions
                A_T_Q_inv_1 *= Ns[n]
            end
            A_Tb_1 += A_T_Q_inv_1*A_1
            a_Tb_1 += A_T_Q_inv_1*y[1,n]
        end
        x_1 = A_Tb_1\a_Tb_1
        T[1] = T_ref(x_1[ii_T_1])
        b[1] = x_1[ii_b_1]

        # T and b estimation
        @threads for k = 2:size(y,1)
            A_Tb, a_Tb = zero(SMatrix{12,12, TT}), zero(SVector{12, TT})
            for n = 1:size(y,2)
                A = J_Tb(u[k,n])
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

        # Normalize Tg
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

        # Logp
        logp = nlogp(y, u, T, b, Q_inv, Ns)
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


end # module
