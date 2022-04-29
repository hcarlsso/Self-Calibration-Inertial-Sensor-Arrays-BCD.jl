show_every = 200
N_bcd  = 100_000
tol_dlogp = 1.0e-11


trace_log_p_2 = zeros(N_bcd)

r_hat, T_hat, b_hat = @time bcd(
    r_hat, T_hat, b_hat, y_s, y_d, Q_inv, Ns, Na , Ng, g_mag, tol_dlogp, N_bcd,
    1.0e-12, 100, 1.0e-13, 200, pool;
    # callback = callback
    callback = get_callback_trace(trace_log_p_2, trace_p, show_every) #callback
)

trace_log_p_2 = trace_log_p_2[trace_log_p_2 .> 0]

trace_log_p = vcat(trace_log_p, trace_log_p_2)

for k = 2:length(trace_p[:k])
    if trace_p[:k][k-1] > trace_p[:k][k]
        trace_p[:k][k] += trace_p[:k][k-1]
    end
end

trace_p_cat = [f(k,v) for (k,v) in trace_p] |> Dict
