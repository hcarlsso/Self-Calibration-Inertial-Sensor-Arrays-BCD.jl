module BcdV3

using StaticArrays
using LinearAlgebra
using Printf
using Base.Threads

export
    bcd!,
    bcd,
    get_callback,
    get_callback_maxdiff_param,
    Ω,
    inverse_affine_Q,
    inverse_affine,
    dynamics


function nlogp(y::Matrix{SVector{3,type}}, u, T, b, Q_inv, Ns) where {type}
    # C = Atomic{type}(0.0)
    N = size(y,2)
    N_triads = size(y,1)
    N_orientions = length(Ns)
    e2 = zeros(type, N_triads, N)
    @threads for n = 1:N
        for k = 1:N_triads
            e = T[k]*u[k,n] + b[k] - y[k,n]
            c_k_n = dot(e, Q_inv[k], e)
            if n <= N_orientions
                c_k_n *= Ns[n]
            end
            # C += c_k_n
            # atomic_add!(C,c_k_n)
            e2[k,n] = c_k_n
        end
    end

    return sum(e2)
end
function nlogp(y::AbstractMatrix{type}, u, T, b, Q_inv, Ns) where {type}

    N = size(y,2)
    inds = reshape(1:size(y,1),3,:)
    N_triads = size(inds,2)
    N_orientions = length(Ns)
    e2 = zeros(type, N_triads, N)
    for n = 1:N
        for k = 1:N_triads
            kk = inds[:,k]
            e = T[:,:,k]*u[kk,n] + b[:,k] - y[kk,n]
            c_k_n = dot(e, Q_inv[kk,kk], e)
            if n <= N_orientions
                c_k_n *= Ns[n]
            end
            e2[k,n] = c_k_n
        end
    end

    return sum(e2)
end
function nlogp_naive(y::Matrix{SVector{3,type}}, Q_inv, Ns, r, T, b, g, w, w_dot, s) where {type}
    C = zeros(type, size(y))
    Na = length(r)
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
                    u = Ω(w[n_d])^2*r[k] + Ω(w_dot[n_d])*r[k] + s[n_d]
                else
                    u = w[n_d]
                end
            end

            e = T[k]*u + b[k] - y[k,n]
            c_k_n = dot(e, Q_inv[k], e)

            if n <= N_orientions
                c_k_n *= Ns[n]
            end

            C[k,n] = c_k_n
        end
    end
    return sum(C)
end
function nlogp_naive(y::Matrix{type}, Q_inv::Matrix{type}, Ns, r::Matrix{type}, T, b::Matrix{type}, g::Matrix{type}, w::Matrix{type}, w_dot::Matrix{type}, s::Matrix{type}) where {type}

    inds = reshape(1:size(y,1), 3, :)
    Na = size(r, 2)
    N = size(y, 2)
    N_triads = size(inds,2)
    N_orientions = length(Ns)
    C = zeros(type,N_triads, N)
    for n = 1:N
        for k = 1:N_triads
            kk = inds[:,k]
            if n <= N_orientions
                if k <= Na
                    u = -g[:,n]
                else
                    u = zeros(type, 3)
                end
            else
                n_d = n - N_orientions
                if k <= Na
                    u = Ω(w[:,n_d])^2*r[:,k] + Ω(w_dot[:,n_d])*r[:,k] + s[:,n_d]
                else
                    u = w[:,n_d]
                end
            end

            e = T[:,:,k]*u + b[:,k] - y[kk,n]
            c_k_n = dot(e, Q_inv[kk,kk], e)

            if n <= N_orientions
                c_k_n *= Ns[n]
            end

            C[k,n] = c_k_n
        end
    end
    return sum(C)
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
function Ω(a::AbstractVector{T}) where {T}
    A = zeros(T,3,3)

    A[1,2] = -a[3]
    A[2,1] =  a[3]

    A[1,3] =  a[2]
    A[3,1] = -a[2]

    A[2,3] = -a[1]
    A[3,2] =  a[1]

    return A
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
function J_Tb(u::AbstractVector{T}) where {T}
    A = zeros(T,3,12)

    for i = 1:3, j = 1:3
        A[j,j + 3(i-1)] = u[i]
    end
    for i = 1:3
        A[i,i+9] = one(T)
    end
    A
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
function J_Tb_ref(u::AbstractVector{T}) where {T}
    A = zeros(T,3,9)
    A[1,1] = u[1]

    A[1,2] = u[2]
    A[2,3] = u[2]

    A[1,4] = u[3]
    A[2,5] = u[3]
    A[3,6] = u[3]

    A[1,7] = one(T)
    A[2,8] = one(T)
    A[3,9] = one(T)
    A
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
function T_ref(x::AbstractVector{TT}) where {TT}
    T = zeros(TT,3,3)
    T[1,1] = x[1]
    T[1,2] = x[2]
    T[1,3] = x[4]

    T[2,2] = x[3]
    T[2,3] = x[5]

    T[3,3] = x[6]
    T
end
function H!(H::AbstractMatrix{T}, r::Vector{SVector{3,T}}) where {T}
    Na = length(r)
    for k = 1:Na
        H[1+3(k-1):3*k,1:3] = -Ω(r[k])
    end
    nothing
end
function H!(H::AbstractMatrix{T}, r::AbstractMatrix{T}) where {T}
    Na = size(r,2)
    inds = reshape(1:length(r), 3,:)
    for k = 1:Na
        kk = inds[:,k]
        H[kk,1:3] = -Ω(r[:,k])
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
function fill_H_constants_2!(H::AbstractMatrix{T}, Na) where {T}
    for k = 1:Na
        H[1+3(k-1):3*k, 4:6] = Matrix{T}(I,3,3)
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
function inverse_affine(y, T, b::Vector{SVector{3,TT}}) where {TT}
    u = [similar_type(y[k,n]) for k = 1:size(y,1), n = 1:size(y,2)]
    inverse_affine!(u, y, T, b)
    return u
end
function inverse_affine!(u, y, T, b::Vector{SVector{3,TT}}) where {TT}
    @threads for n = 1:size(y,2)
        for k = 1:size(y,1)
            u[k,n] = T[k]\(y[k,n] - b[k])
        end
    end
    nothing
end
function inverse_affine!(u, y, T, b)
    inds = reshape(1:size(y,1), 3,:)
    for k = 1:size(inds,2)
        kk = inds[:,k]
        u[kk,:] = T[:,:,k]\(y[kk,:] .- b[:,k])
    end
    nothing
end
function inverse_affine_Q(Q_inv, T::Vector{SMatrix{3,3,TT}}) where {TT}
    Qu_inv = [similar_type(Q_k) for Q_k in Q_inv]
    inverse_affine_Q!(Qu_inv, Q_inv, T)
    return Qu_inv
end
function inverse_affine_Q!(Qu_inv, Q_inv, T::Vector{SMatrix{3,3,TT,9}}) where {TT}
    for n = 1:length(Q_inv)
        Qu_inv[n] = T[n]'*Q_inv[n]*T[n]
    end
    nothing
end
function inverse_affine_Q!(Qu_inv::AbstractMatrix{TT}, Q_inv::AbstractMatrix{TT}, T::AbstractArray{TT,3}) where {TT}
    inds = reshape(1:size(Q_inv,1), 3,:)
    for k = 1:size(inds,2)
        kk = inds[:,k]
        Qu_inv[kk,kk] = T[:,:,k]'*Q_inv[kk,kk]*T[:,:,k]
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
function gravity!(g, u, Qu_inv, g_mag::T, tol, i_max) where {T}
    inds = reshape(1:size(u,1),3,:)
    Na = size(inds,2)
    D = mapreduce(k -> Qu_inv[inds[:,k], inds[:,k]], +, 1:Na)
    sigs = svdvals(D)
    (sig_max, sig_min) = sigs[1], sigs[3]
    for n = 1:size(u,2)
        d = mapreduce(k -> Qu_inv[inds[:,k], inds[:,k]]*u[inds[:,k],n], +, 1:Na)
        g[:,n] = estimate_gravity(SVector{3, T}(d), SMatrix{3,3,T}(D), sig_min, sig_max, g_mag, tol, i_max) |> Array
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
# Gauss Newton
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
"""
    min f(w)
    over w

    Using Gauss Newton

    f(w) = (y - h(w,r))^T * P*(y - h(w,r))
"""
function gauss_newton(w::AbstractVector{T}, y::AbstractVector{T}, P, r, Ng::Int, g_tol, i_max;
                      warn = true) where {T}

    N = length(y)

    H = zeros(T,3,3)   # Hessian
    g = zeros(T,3)     # Gradient
    p = zeros(T,3)     # Step

    J = zeros(T,N,3)
    y_pred = zeros(T,N)
    e = zeros(T,N)

    J_T_P = zeros(T, 3, N)

    Na = size(r,2)
    fill_J_h_buff!(J, Na, Ng)

    k = 0::Int
    converged = false

    while !converged
        k += 1

        h!(y_pred, w, r, Ng)
        e .= y - y_pred

        J_h!(J, w, r, Ng)


        J_T_P .= J' * P
        H .= J_T_P * J
        g .= J_T_P * e

        p .= H \ g
        w += p

        # ||Jp||/(1+||r||
        # Should acutally be  ||RJp||/(1+||Rr||  where R'R = P
        converged = k == i_max || norm(J * p, Inf) <  g_tol * (1.0 + norm(e, Inf))
    end
    if warn && k == i_max
        crit = norm(J * p, Inf) / (1.0 + norm(e, Inf))

        @warn @sprintf(
            "GS: Max iterations %d reached. Criterion ||Jp||/(1+||r||) = %.2e < %.1e = tol \n",
            k, crit, g_tol
        )
        converged = false
    end

    w, converged
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
        u_n[n] = vcat(u[:,n]...)
    end
    w = [copy(w0_n) for w0_n in w0]
    dynamics!!(w, w_dot, s, u_n, Qu_inv_tall, r, Ng, tol_gs, i_max_w)

    return w, w_dot, s
end
function dynamics!!(w::Vector{SVector{3,T}}, w_dot::Vector{SVector{3,T}}, s::Vector{SVector{3,T}},
    u::Vector{SVector{N,T}}, Qu_inv::SMatrix{N,N,T},
    r, Ng, g_tol, i_max) where{N,T}
    N_dynamics = length(w)
    Na = size(r, 2)

    H_m = zero(MMatrix{N, 6, T})
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

    r_m = hcat(r...)
    for n = 1:N_dynamics
        w[n] = gauss_newton(w[n], e, u[n], P, r_m, Ng, g_tol, i_max, y_pred, J_m)
        h!(y_pred, w[n], r_m, Ng)
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
    function callback(k::Int, dx, dlogp, log_p, r, T, b)
        if mod(k,show_every) == 0
            @printf("%9d logp %.18e  dlogp % .5e dx % .5e\n", k, log_p, dlogp, dx)
        end
    end
    return callback
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
function maxdiff(r, T, b, r_p, T_p, b_p)

    dr = (maxdiff(r[k], r_p[k]) for k in eachindex(r))
    dT = (maxdiff(T[k], T_p[k]) for k in eachindex(T))
    db = (maxdiff(b[k], b_p[k]) for k in eachindex(b))

    d = reduce(max, dr)
    d = max(d, reduce(max, dT))
    max(d, reduce(max, db))
end
################################################################################
# BCD loop
################################################################################
function bcd!(r,T,b, y, Q_inv, Ns, g_mag::TT, ::Val{N_sens}, ::Val{Na},
              tol_interval, i_max_g, tol_gs, i_max_w, tol_bcd_dlogp, tol_bcd_dx, i_max_bcd;
              callback = nothing, warn = true, w0 = nothing, verbose = false,
              update_Ta = true, update_Tg = true) where {TT, N_sens, Na}

    display(T[1])
    @assert istriu(T[1])
    @assert r[1] == zeros(3)
    @assert length(T) == length(b) == size(y,1) == length(Q_inv)
    @assert length(r) == Na

    N = size(y, 2)
    N_triads = size(y, 1)
    Ng = N_triads - Na
    N_orientions = length(Ns)
    N_dynamics = N - N_orientions

    println("N dynamics: ", N_dynamics)
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
    if w0 == nothing
        w = map(1:N_dynamics) do n
            mapreduce(k -> y[Na + k, n + N_orientions], +, 1:Ng)/Ng
        end
    else
        w = copy(w0)
    end

    w_dot = [zero(SVector{3,TT}) for _ = 1:N_dynamics]
    s = [zero(SVector{3,TT}) for _ = 1:N_dynamics]

    u_pred_all = [zero(MVector{N_sens, TT}) for _ = 1:N_dynamics]
    # WLS = zero(SMatrix{6, N_sens, TT})
    H_m = zero(MMatrix{N_sens, 6, TT})
    fill_H_constants!(H_m, Na)

    # Buffs
    u_pred = [zero(MVector{N_sens, TT}) for _ = 1:nthreads()]
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

    if update_Ta && update_Tg
        slice = 2:N_triads
    elseif update_Ta && !update_Tg
        slice = 2:Na
    elseif !update_Ta && update_Tg
        slice = Na+1:N_triads
    else
        slice = 2:0
    end

    k = 0
    r_p = copy(r)
    T_p = copy(T)
    b_p = copy(b)
    log_p_prev::TT = Inf
    dlogp = zero(TT)
    logp = zero(TT)
    dx = zero(TT)
    converged = false
    while !converged
        k += 1

        inverse_affine!(u, y, T, b)
        inverse_affine_Q!(Qu_inv, Q_inv, T)

        if verbose
            lc = nlogp_naive(y, Q_inv, Ns, r, T, b, g, w, w_dot,s)
            @printf "1: %.3e\n" lc
            lp = lc
        end

        gravity!(g, u, Qu_inv, Na, N_orientions, g_mag, tol_interval, i_max_g)

        if verbose
            lc = nlogp_naive(y, Q_inv, Ns, r, T, b, g, w, w_dot,s)
            @printf "2: %.3e diff %.3e\n" lc lp-lc
            lp = lc
        end

        # Dynamics
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
            u_pred_i = u_pred[threadid()]
            J_m_i = J_m[threadid()]

            w[n] = gauss_newton(w[n], e, u_n[n], P, r_m, Ng, tol_gs, i_max_w,
                                u_pred_i, J_m_i)
            h!(u_pred_i, w[n], r_m, Ng)
            u_pred_all[n] = copy(u_pred_i)
            e = u_n[n] - SVector(u_pred_i)
            phi = WLS*e
            w_dot[n] = phi[inds_w_dot]
            s[n] = phi[inds_s]
        end

        if verbose
            lc = nlogp_naive(y, Q_inv, Ns, r, T, b, g, w, w_dot,s)
            @printf "3: %.3e diff %.3e\n" lc lp-lc
            lp = lc
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

        if verbose
            lc = nlogp_naive(y, Q_inv, Ns, r, T, b, g, w, w_dot,s)
            @printf "4: %.3e diff %.3e\n" lc lp-lc
            lp = lc
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
        if update_Ta
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
        end
        if verbose
            lc = nlogp_naive(y, Q_inv, Ns, r, T, b, g, w, w_dot,s)
            @printf "5: %.3e diff %.3e\n" lc lp-lc
            lp = lc
        end

        # T and b estimation
        # Dont update T_g if only static measurements
        @threads for k = slice
            A_Tb, a_Tb = zero(SMatrix{12,12, TT}), zero(SVector{12, TT})

            for n = 1:N
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

        if verbose
            lc = nlogp_naive(y, Q_inv, Ns, r, T, b, g, w, w_dot,s)
            @printf "6: %.3e diff %.3e\n" lc lp-lc
            lp = lc
        end

        # Normalize Tg
        if update_Tg
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

        if verbose
            lc = nlogp_naive(y, Q_inv, Ns, r, T, b, g, w, w_dot,s)
            @printf "7: %.3e diff %.3e\n" lc lp-lc
            lp = lc
        end

        # Logp
        logp = nlogp(y, u, T, b, Q_inv, Ns)
        dlogp = log_p_prev - logp
        log_p_prev = logp

        dx = maxdiff(r, T, b, r_p, T_p, b_p)

        r_p .= r
        T_p .= T
        b_p .= b

        if callback != nothing
            callback(k, dx, dlogp, logp, r, T, b)
        end

        converged = k == i_max_bcd || abs(dlogp) < tol_bcd_dlogp || dx < tol_bcd_dx
    end
    if k == i_max_bcd && warn
        @warn join(
        [@sprintf("BCD: Max iterations %d reached", k),
         @sprintf("Criterion |dlopp| = %.2e < %.1e = tol",abs(dlogp), tol_bcd_dlogp),
         @sprintf("Criterion ||dx||_∞ = %.2e < %.1e = tol", dx, tol_bcd_dx)],
        "\n"
        )
    end
    eta = Dict(
        :w => w,
        :w_dot => w_dot,
        :s => s,
        :g => g
    )
    return r, T, b, eta
end

function bcd(r, T, b, y, Q_inv, Ns, g_mag,
    tol_interval, i_max_g, tol_gs, i_max_w, tol_bcd_dlogp, tol_bcd_dx, i_max_bcd;
    kwargs...)

    TT = Float64

    Na = size(r,2)
    Nt = size(T,3)

    rs = [SVector{3,TT}(r[:,k]) for k = 1:Na]
    bs = [SVector{3,TT}(b[:,k]) for k = 1:Nt]
    Ts = [SMatrix{3,3,TT}(T[:,:,k]) for k = 1:Nt]

    ys = [SVector{3,TT}(y[1+3(k-1):3*k, n]) for k = 1:Nt, n = 1:size(y,2)]
    Q_inv_s = [SMatrix{3,3,TT}(Q_inv[:,:,k]) for k = 1:Nt]

    r_hat, T_hat, b_hat, eta_hat = bcd!(rs, Ts, bs, ys, Q_inv_s, Ns, g_mag,
                                        Val{3*Nt}(),Val{Na}(),
                                        tol_interval, i_max_g,
                                        tol_gs, i_max_w,
                                        tol_bcd_dlogp, tol_bcd_dx, i_max_bcd;
                                        kwargs...)

    eta = map(eta_hat |> collect) do (k,v)
        a = hcat(Array.(v)...)
        (k,a)
    end |> Dict
    theta = Dict(
        :r => hcat(Array.(r_hat)...),
        :b => hcat(Array.(b_hat)...),
        :T => cat(Array.(T_hat)...; dims = 3)
    )

    theta, eta
end
function bcd!(r::AbstractMatrix{type}, T::AbstractArray{type,3}, b::AbstractMatrix{type},
              y::AbstractMatrix{type}, Q_inv::AbstractMatrix{type},
              Ns, g_mag::type,
              tol_interval, i_max_g, tol_gs, i_max_w, tol_bcd_dlogp, tol_bcd_dx, i_max_bcd;
              callback = nothing, warn = true, w0 = nothing,
              update_Ta = true, update_Tg = true,
              verbose = false) where {type}

    N_sens = size(y,1)
    Na = size(r,2)
    N_triads = Int(N_sens/3)
    Ng =  N_triads - Na

    @assert istriu(T[:,:,1])
    @assert r[:,1] == zeros(3)
    @assert size(y,1) == size(Q_inv,1) == size(Q_inv,2) == length(b)




    N = size(y,2)
    N_orientions = length(Ns)
    N_dynamics = N - N_orientions
    if verbose
        println("N dynamics: ", N_dynamics)
        println("N orientations: ", N_orientions)
    end

    inds = reshape(1:3*N_triads, 3, :)

    # Initial w0 from gyroscopes
    if w0 == nothing
        @assert Ng > 0
        y_g = y[3*Na + 1:end, N_orientions + 1:end]

        # Mean value
        w = map(1:3) do k
            sum(y_g[k:3:end, :]; dims = 1)/Ng
        end |> x -> vcat(x...)
    else
        w = copy(w0)
    end
    w_dot = zeros(type,3, N_dynamics)
    s = zeros(type,3, N_dynamics)
    g = zeros(type,3, N_orientions)

    u = zeros(type, size(y))
    Qu_inv = zeros(type, size(Q_inv))

    H = zeros(type, N_sens, 6)
    fill_H_constants_2!(H, Na)

    Jrs = zeros(type, 3, 3, N_dynamics)
    u_dyn_pred = zeros(type,3*N_triads, N_dynamics)

    if update_Ta && update_Tg
        slice = 2:N_triads
    elseif update_Ta && !update_Tg
        slice = 2:Na
    elseif !update_Ta && update_Tg
        slice = Na+1:N_triads
    else
        slice = 2:0
    end

    # Convergence stuff
    k = 0
    theta_p = vcat(r[:], T[:], b[:])
    log_p_prev::type = Inf
    dlogp = zero(type)
    logp = zero(type)
    dx = zero(type)
    converged = false
    converged_bisection = zeros(Bool, N_orientions, i_max_bcd)
    converged_gauss_newton = zeros(Bool, N_dynamics, i_max_bcd)

    while !converged
        k += 1
        if verbose
            println("BCD iteration: $(k)")
        end

        inverse_affine!(u, y, T, b)
        inverse_affine_Q!(Qu_inv, Q_inv, T)

        if verbose
            lc = nlogp_naive(y, Q_inv, Ns, r, T, b, g, w, w_dot,s)
            @printf "1: %.3e\n" lc
            lp = lc
        end

        gravity!(g, u[1:3*Na,1:N_orientions], Qu_inv, g_mag, tol_interval, i_max_g)

        if verbose
            lc = nlogp_naive(y, Q_inv, Ns, r, T, b, g, w, w_dot,s)
            @printf "2: %.3e diff %.3e\n" lc lp-lc
            lp = lc
        end

        # Dynamics
        H!(H, r)
        # Since H is zero for gyro scope measurements
        # P = Q_inv - Q_inv*H*(H'*Q_inv*H)^{-1}*H'*Q_inv
        WLS = (H'*Qu_inv*H)\(H'*Qu_inv)
        P = Qu_inv  - Qu_inv*H*WLS
        u_dyn = u[:,1 + N_orientions:end]
        @threads for n = 1:N_dynamics

            if verbose
                println("GS iteration: $(n)")
            end

            w[:,n], converged_gauss_newton[n,k] = gauss_newton(
                w[:,n], u_dyn[:,n], P, r, Ng, tol_gs, i_max_w;
                warn = warn
            )
            if !converged_gauss_newton[n,k]
                @warn @sprintf("BCD Iteration %d, time sample %d did not converge in GS", k, n)
            end

            @views h!(u_dyn_pred[:,n], w[:,n], r, Ng)
        end

        phi = WLS*(u_dyn - u_dyn_pred)
        w_dot[:,:] = phi[1:3,:]
        s[:,:] = phi[4:6,:]

        if verbose
            lc = nlogp_naive(y, Q_inv, Ns, r, T, b, g, w, w_dot,s)
            @printf "3: %.3e diff %.3e\n" lc lp-lc
            lp = lc
        end

        # Positions
        @threads for n = 1:N_dynamics
            Jrs[:,:,n] = J_r(w[:,n], w_dot[:,n])
        end
        @threads for k = 2:Na
            kk = inds[:,k]
            A_r, a_r = zeros(type,3,3), zeros(type,3)
            for n = 1:N_dynamics
                J = Jrs[:,:,n]'*Qu_inv[kk,kk]
                A_r += J*Jrs[:,:,n]
                a_r += J*(u[kk,n + N_orientions] - s[:,n])
            end
            r[:,k] = A_r\a_r
        end

        if verbose
            lc = nlogp_naive(y, Q_inv, Ns, r, T, b, g, w, w_dot,s)
            @printf "4: %.3e diff %.3e\n" lc lp-lc
            lp = lc
        end
        # Update u
        @threads for n = 1:N
            for k = 1:N_triads
                kk = inds[:,k]
                # Static
                if n <= N_orientions
                    if k <= Na
                        u[kk,n] = -g[:,n]
                    else
                        u[kk,n] = zeros(type,3)
                    end
                else
                    n_d = n - N_orientions
                    if k <= Na
                        u[kk,n] = Jrs[:,:,n_d]*r[:,k] + s[:,n_d]
                    else
                        u[kk,n] = w[:,n_d]
                    end
                end
            end
        end

        # Ref Tb
        if update_Ta
            A_Tb_1, a_Tb_1 = zeros(type,9,9), zeros(type,9)
            for n = 1:N
                A_1 = J_Tb_ref(u[1:3,n])
                A_T_Q_inv_1 = A_1'*Q_inv[1:3,1:3]
                if n <= N_orientions
                    A_T_Q_inv_1 *= Ns[n]
                end
                A_Tb_1 += A_T_Q_inv_1*A_1
                a_Tb_1 += A_T_Q_inv_1*y[1:3,n]
            end
            x_1 = A_Tb_1\a_Tb_1
            T[:,:,1] = T_ref(x_1[1:6])
            b[:,1] = x_1[7:9]
        end
        if verbose
            lc = nlogp_naive(y, Q_inv, Ns, r, T, b, g, w, w_dot,s)
            @printf "5: %.3e diff %.3e\n" lc lp-lc
            lp = lc
        end

        # T and b estimation
        @threads for k = slice
            A_Tb, a_Tb = zeros(type,12,12), zeros(type,12)
            kk = inds[:,k]
            for n = 1:N
                A = J_Tb(u[kk,n])
                A_T_Q_inv = A'*Q_inv[kk,kk]
                if n <= N_orientions
                    A_T_Q_inv *= Ns[n]
                end
                A_Tb += A_T_Q_inv*A
                a_Tb += A_T_Q_inv*y[kk,n]
            end

            x = A_Tb\a_Tb
            T[:,:,k] = reshape(x[1:9], 3,3)
            b[:,k] = x[10:12]
        end

        if verbose
            lc = nlogp_naive(y, Q_inv, Ns, r, T, b, g, w, w_dot,s)
            @printf "6: %.3e diff %.3e\n" lc lp-lc
            lp = lc
        end
        # Normalize Tg
        if update_Tg
            C::type = 0.0
            for k = 1:Ng
                C += tr(T[:,:,k+Na])
            end
            C /= (3*Ng)
            for k = 1:Ng
                T[:,:,k+Na] ./= C
            end
        end

        if verbose
            lc = nlogp_naive(y, Q_inv, Ns, r, T, b, g, w, w_dot,s)
            @printf "7: %.3e diff %.3e\n" lc lp-lc
            lp = lc
        end

        # Logp
        logp = nlogp(y, u, T, b, Q_inv, Ns)
        dlogp = log_p_prev - logp
        log_p_prev = logp
        theta = vcat(r[:], T[:], b[:])
        dx = maxdiff(theta, theta_p)

        theta_p .= theta

        if callback != nothing
            params = Dict(
                :r => r,
                :T => T,
                :b => b,
                :w => w,
                :w_dot => w_dot,
                :s => s,
                :g => g,
            )
            callback(k, dx, dlogp, logp, params)
        end

        converged = k == i_max_bcd || abs(dlogp) < tol_bcd_dlogp || dx < tol_bcd_dx
    end
    if k == i_max_bcd && warn
        @warn join(
            [@sprintf("BCD: Max iterations %d reached", k),
             @sprintf("Criterion |dlopp| = %.2e < %.1e = tol",abs(dlogp), tol_bcd_dlogp),
             @sprintf("Criterion ||dx||_∞ = %.2e < %.1e = tol", dx, tol_bcd_dx)],
            "\n"
        )
    end
    theta = Dict(
        :r => r,
        :T => T,
        :b => b,
    )
    eta = Dict(
        :w => w,
        :w_dot => w_dot,
        :s => s,
        :g => g,
    )
    convergence_assessment = Dict(
        :gs_iterations => converged_gauss_newton
    )
    return theta, eta, convergence_assessment
end


end # module
