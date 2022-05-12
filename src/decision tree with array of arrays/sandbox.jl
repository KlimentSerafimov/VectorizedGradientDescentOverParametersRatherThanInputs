function dt(p, x)
    if(x < p[1])
        if (x < p[2])
            return p[3]
        else
            return p[4]
        end
    else
        if(x < p[5])
            return p[6]
        else
            return p[7]
        end
    end
end

dt_loss(p, x, y) = (dt(p, x) - y)^2

function dt_harness(p, data)
    n = length(data)
    error = 0
    for i in 1:n
        error += dt_loss(p, data[i][1], data[i][2])
    end
    return error
end

data = [[0, 0], [1/4, 1/4], [2/4, 2/4], [3/4, 3/4]]
dt_wrapper(p) = dt_harness(p, data)

using ForwardDiff
function gradient_descent(f, p)
    dfdp(_p) = ForwardDiff.gradient(f, _p)
    cutoff = 0.0001
    alpha = 0.01
    delta = 1
    error = f(p)
    while(error > cutoff && delta > cutoff/10)
        prev_error = error
        p = p - alpha*dfdp(p)
        error = f(p)
        delta = prev_error-error
    end
    return error, p
end

MyFloat = Float64
using ThreadsX

function multi_gradient_descent(f, n, m)
    errors = Array{Tuple{MyFloat, Int64}}(undef, m);
    ps = Array{Array{MyFloat}}(undef, m);

#     Threads.@threads
    for i in 1:m
        p = rand(MyFloat, n)
        error, p = gradient_descent(f, p)
        errors[i] = (error, i)
        ps[i] = p
    end

#     min_error_and_i = ThreadsX.min(all_final_errors...)
    min_error_and_i = min(errors...)

    best_error = min_error_and_i[1]
    best_p = ps[min_error_and_i[2]]

    return best_error, best_p
end

function sampling_search(f, n, m)
    final_error = typemax(MyFloat)
    final_p = undef
    for i in 1:m
        p = rand(MyFloat, n);
        error = f(p);
        if final_error > error
            final_error = error
            final_p = p
        end
    end
    return final_error, final_p
end


#---------DRIVERS----------

function driver_dt_wrapper()
    n = 7
    init_p = rand(MyFloat, n)

    println("init_p ", init_p)

    init_error = dt_wrapper(init_p)
    println("init_error ", init_error)
end

function driver_gradient_descent()
    n = 7
    init_p = rand(MyFloat, n)

    final_error, final_p = gradient_descent(dt_wrapper, init_p)
    println("final_error ", final_error)
    println("final_p ", final_p)
end

function driver_multi_gradient_descent()
    n = 7
    m = 100

    final_error, final_p = multi_gradient_descent(dt_wrapper, n, m)

    println("final_error ", final_error)
    println("final_p ", final_p)
end

function plot_success_rate_vs_num_trials()
    n = 7
    meta_samples = 100
    success_rates = []
    avg_errors = []
    ms = 5:5:140
    for m in ms
        function run_meta_trials(meta_trials)
            cutoff = 0.001
            successes = 0
            sum_error = 0
            for i in 1:meta_trials
                final_error, final_p = multi_gradient_descent(dt_wrapper, n, m)
                successes += final_error < cutoff;
                sum_error += final_error
            end
            return successes/meta_trials, final_error/meta_trials
        end

        t = @elapsed success_rate, avg_error = run_meta_trials(meta_samples)

        push!(success_rates, success_rate)
        push!(avg_errors, avg_error)

        println("m = ", m, " success_rate = ", success_rate, "avg_error = ", avg_error, " t = ", t)

    end

    p = plot(title = "Number of trials vs success rate",
    xaxis = "m (number of trails)", yaxis = "success rate",legend=:topleft)
    plot!(p, ms, success_rates, label = "success rate", linewidth=2)
    display(p)

#     readline()
end

# plot_success_rate_vs_num_trials()

function plot_avg_error_vs_num_samples()
    n = 7
    meta_samples = 100
    success_rates = []
    avg_errors = []
    ms = [10^x for x in 0:6]
    for m in ms
        function run_meta_trials(meta_trials)
            cutoff = 0.001
            successes = 0
            sum_errors = 0
            for i in 1:meta_trials
                final_error, final_p = sampling_search(dt_wrapper, n, m)
                successes += final_error < cutoff;
                sum_errors += final_error
            end
            return successes/meta_trials, sum_errors/meta_trials
        end

        t = @elapsed success_rate, avg_error = run_meta_trials(meta_samples)

        push!(success_rates, success_rate)
        push!(avg_errors, avg_error)

        println("m = ", m, " success_rate = ", success_rate, " avg_error = ", avg_error, " t = ", t)
    end

    p = plot(title = "Number of numples vs average error",
    xaxis = "m (number of numples)", yaxis = "average error",legend=:topleft)
    plot!(p, ms, success_rates, label = "average error", linewidth=2)
    display(p)

#     readline()
end

# plot_avg_error_vs_num_samples()
# import Pkg; Pkg.add("PlotlyJS")
# using PlotlyJS

# using Plots

using PlotlyJS

function meta_sampler(method, ms, title, is_x_axis_log)
    n = 7
    meta_samples = 1000
    success_rates = []
    avg_errors = []
    ts = []
    method(dt_wrapper, n, ms[1])
    for m in ms
        function run_meta_trials(meta_trials)
            cutoff = 0.001
            successes = 0
            sum_errors = 0
            for i in 1:meta_trials
                final_error, final_p = method(dt_wrapper, n, m)
                successes += final_error < cutoff;
                sum_errors += final_error
            end
            return successes/meta_trials, sum_errors/meta_trials
        end

        t = @elapsed success_rate, avg_error = run_meta_trials(meta_samples)

        push!(success_rates, success_rate)
        push!(avg_errors, avg_error)
        push!(ts, t/meta_samples*1000)

        println("m = ", m, " success_rate = ", success_rate, " avg_error = ", avg_error, " t = ", t/meta_samples*1000)

    end

#     xaxis_params = "m (#trails)"
#     if is_x_axis_log
#         xaxis_params = (xaxis_params, :log)
#     end
#     p = Plots.plot(ms, avg_errors, title = title * "\n#trials vs success rate and average error",
#     xaxis = xaxis_params, yaxis = "success rate \\/ average error", linewidth=2, label = "average error", legend_font_pointsize = 12)
#     Plots.plot!(ms, success_rates, label = "success rate",legend=:bottomleft, linewidth=2)
#     Plots.display(p)

    line_width = 4

    PlotlyJS.plot(
        [
            PlotlyJS.scatter(x=ms, y=success_rates, name="success rate", yaxis = "y1", line_color = "red", line_width = line_width),
            PlotlyJS.scatter(x=ms, y=avg_errors, name="average error", yaxis="y2", line_color = "blue", line_width = line_width),
            PlotlyJS.scatter(x=ms, y=ts, name="runtime (ms)", yaxis="y3", line_color = "green", line_width = line_width)
        ],
        Layout(
            font_size=32,
            xaxis_domain=[0.15, 0.95],
            yaxis_domain=[0.0, 0.95],
            yaxis=attr(title="success rate", titlefont_color="red", position = 0.15),
            yaxis2=attr(
                title="average error", titlefont_color="blue",
                overlaying="y", side="left", position=0.05, anchor="free"
            ),
            yaxis3=attr(
                title="runtime (ms)", titlefont_color="green",
                overlaying="y", side="right", anchor="x",
            ),
            title_text=title * "<br>#trials vs success rate, average error, and runtime",
            xaxis_title = "m (#trials)"
        )
    )


end

function driver_meta_sampler()
#     meta_sampler(sampling_search, [ceil(sqrt(10)^x) for x in 0:12], "Sampling Search", true)
    meta_sampler(multi_gradient_descent, 5:5:140, "Multi Gradient Descent Search", false)
end

driver_meta_sampler()

#=
sandbox:
- Julia version: 1.7.2
- Author: kliment
- Date: 2022-05-06
=#


"""
function driver_meta_sampler()
#     meta_sampler(sampling_search, [ceil(sqrt(10)^x) for x in 0:12], "Sampling Search", true)
    meta_sampler(multi_gradient_descent, 5:5:140, "Multi Gradient Descent Search", false)
end
m = 5 success_rate = 0.082 avg_error = 0.033405738302363826 t = 1.006858098
m = 10 success_rate = 0.156 avg_error = 0.026788861721320036 t = 2.012076298
m = 15 success_rate = 0.224 avg_error = 0.02450642692385003 t = 3.035492649
m = 20 success_rate = 0.243 avg_error = 0.02387873133313709 t = 4.061893922
m = 25 success_rate = 0.338 avg_error = 0.020910493894900097 t = 5.522496555
m = 30 success_rate = 0.381 avg_error = 0.01956539589932681 t = 6.71077592
m = 35 success_rate = 0.408 avg_error = 0.01872077970458178 t = 7.644175696
m = 40 success_rate = 0.48 avg_error = 0.016471280771752275 t = 8.13101446
m = 45 success_rate = 0.503 avg_error = 0.0157525201021167 t = 8.945542393
m = 50 success_rate = 0.557 avg_error = 0.014064556261699078 t = 9.971280884
m = 55 success_rate = 0.556 avg_error = 0.014094626542250785 t = 10.95988537
m = 60 success_rate = 0.607 avg_error = 0.012503033485324926 t = 11.987331624
m = 65 success_rate = 0.666 avg_error = 0.010661787747304445 t = 12.965347293
m = 70 success_rate = 0.657 avg_error = 0.01094041600242995 t = 14.267726384
m = 75 success_rate = 0.706 avg_error = 0.009410636760251634 t = 15.855546175000002
m = 80 success_rate = 0.714 avg_error = 0.009159872834701176 t = 16.727627082
m = 85 success_rate = 0.743 avg_error = 0.008254741451910515 t = 18.224589737
m = 90 success_rate = 0.758 avg_error = 0.007787189149179354 t = 19.302972134
m = 95 success_rate = 0.76 avg_error = 0.007725040234031046 t = 20.984456476
m = 100 success_rate = 0.778 avg_error = 0.0071609438817417876 t = 22.257334136
m = 105 success_rate = 0.801 avg_error = 0.006444384733447451 t = 22.305553588
m = 110 success_rate = 0.831 avg_error = 0.005508015109510392 t = 23.21286592
m = 115 success_rate = 0.838 avg_error = 0.0052895778020456185 t = 24.067945453
m = 120 success_rate = 0.842 avg_error = 0.005164506003235521 t = 25.045624339
m = 125 success_rate = 0.878 avg_error = 0.00404146274222388 t = 26.01443451
m = 130 success_rate = 0.88 avg_error = 0.00397870796775774 t = 27.041151521
m = 135 success_rate = 0.894 avg_error = 0.003541980495434792 t = 28.202800003
m = 140 success_rate = 0.874 avg_error = 0.004165901306803361 t = 29.428625206
"""