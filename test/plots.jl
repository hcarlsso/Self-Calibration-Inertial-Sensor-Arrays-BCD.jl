let
    fig,  ax = subplots(1,1)
    ax.plot(1:length(trace_log_p), trace_log_p)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(true)

end
let
    fig,  ax = subplots(1,1)
    -diff(trace_log_p) |> x -> ax.plot(1:length(x), x)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(true)
    ax.set_title("diff p")
end

plot_trace_value(p, ii) = begin
    fig,  ax = subplots(1,1)
    ax.plot(trace_p_cat[:k], trace_p_cat[p][ii,:])
    ax.plot(trace_p_cat[:k], p_true[p][ii]*ones(length(trace_p_cat[:k])), "-r")
    ax.grid(true)
    ax.set_xscale("log")
    ax.set_title("Trace $(p) index $(ii)")
end

plot_error(p, ii) = begin
    fig,  ax = subplots(1,1)
    ax.plot(trace_p_cat[:k], trace_p_cat[p][ii,:] .- p_true[p][ii] .|> abs)
    ax.grid(true)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Error $(p) index $(ii)")
end

parameters_2_plot =[
    (:T, CartesianIndex(1,1,1)),
    (:T, CartesianIndex(1,1,9)),
    (:T, CartesianIndex(1,2,9)),
    (:r, CartesianIndex(2,2)),
]
for (p, ii) in parameters_2_plot
    plot_error(p,ii)
    plot_trace_value(p, ii)
end
