using Random

Random.seed!(1)

function get_callback_trace(trace, trace_p, show_every = 1)
    function callback(k::Int, dx, dlogp, log_p, p)
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
            push!(trace_p[:r], p[:r])
            push!(trace_p[:T], p[:T])
            push!(trace_p[:b], p[:b])
            push!(trace_p[:k], k)
        end
    end
    return callback
end

r0_m = hcat(r0...) |> Array
T0_m = cat(T0...;dims = 3) |> Array
b0_m = hcat(b0...) |> Array
y_m = reshape(vcat((y .|> Array )...), :,size(y,2))
Q_inv_m = cat((Q_inv .|> Array)...;dims = (1,2))

show_every = 1
# N_bcd  = 10
tol_dlogp = 1.0e-9

trace = Dict(
    :f => zeros(N_bcd),
    :f_abs => zeros(N_bcd),
    :f_rel => zeros(N_bcd),
    :x_abs => zeros(N_bcd)
)
trace_p = Dict(:r => [], :T => [], :b => [], :k => [])
r_hat_m = copy(r0_m)
T_hat_m = copy(T0_m)
b_hat_m = copy(b0_m)

theta_m, eta_m, conv = bcd!(
    r_hat_m, T_hat_m, b_hat_m, y_m, Q_inv_m, Ns, g_mag,
    1.0e-12, 100, 1.0e-13, 200, tol_dlogp, 1.0-14, N_bcd;
    callback = get_callback_trace(
        trace, trace_p, show_every
    )
)

# end
