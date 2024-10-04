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

function run_meta_trials(method, n, m, meta_samples)
    cutoff = 0.001
    successes = 0
    sum_errors = 0
    for i in 1:meta_samples
        final_error, final_p = method(dt_wrapper, n, m)
        successes += final_error < cutoff;
        sum_errors += final_error
    end
    return successes/meta_samples, sum_errors/meta_samples
end

function meta_sampler(method, ms, title)
    println("ms ", ms)
    n = 7
    meta_samples = 1000
    println("meta_samples ", meta_samples)
    success_rates = []
    avg_errors = []
    ts = []
    run_meta_trials(method, n, ms[1], meta_samples)
    for m in ms

        t = @elapsed success_rate, avg_error = run_meta_trials(method, n, m, meta_samples)

        push!(success_rates, success_rate)
        push!(avg_errors, avg_error)
        push!(ts, t/meta_samples*1000)

        println("m = ", m, " success_rate = ", success_rate, " avg_error = ", avg_error, " t(ms) = ", t/meta_samples*1000)

    end

    println("ms ", ms)
    println("success_rates ", success_rates)
    println("avg_errors ", avg_errors)
    println("ts ", ts)

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
            xaxis_type="log",
            yaxis=attr(
                title="success rate", titlefont_color="red", position = 0.15,
                ),
            yaxis2=attr(
                title="average error", titlefont_color="blue",
                overlaying="y", side="left", position=0.05, anchor="free",
            ),
            yaxis3=attr(
                title="runtime (ms)", titlefont_color="green",
                overlaying="y", side="right", anchor="x",
                type = "log",
                range=[0, 2]
            ),
            title_text=title * "<br>#trials vs success rate, average error, and runtime",
            xaxis_title = "m (#trials)"
        )
    )

end

function hybrid_search(f, n, m)
    best_error, best_p = sampling_search(f, n, m)
    best_error, best_p = gradient_descent(f, best_p)
    return best_error, best_p
end


#---------DRIVERS----------

function driver_dt_wrapper()
    n = 7
    init_p = rand(MyFloat, n)

    println("init_p ", init_p)

    init_error = dt_wrapper(init_p)
    println("init_error ", init_error)
end

# driver_dt_wrapper()

function driver_gradient_descent()
    n = 7
    init_p = rand(MyFloat, n)

    final_error, final_p = gradient_descent(dt_wrapper, init_p)
    println("final_error ", final_error)
    println("final_p ", final_p)
end

driver_gradient_descent()

function driver_multi_gradient_descent()
    n = 7
    m = 100

    final_error, final_p = multi_gradient_descent(dt_wrapper, n, m)

    println("final_error ", final_error)
    println("final_p ", final_p)
end

using PlotlyJS

# meta_sampler(sampling_search, [trunc(Int, (ceil(100*(sqrt(10)^x)))) for x in 0:9], "Sampling Search")
# [100, 317, 1001, 3163, 10001, 31623, 100001, 316228, 1000001, 3162278]
# m = 100 success_rate = 0.0 avg_error = 0.06937220969827483 t = 0.028327790000000002
# m = 317 success_rate = 0.0 avg_error = 0.05395262138616602 t = 0.08155611
# m = 1001 success_rate = 0.0 avg_error = 0.04230030205385441 t = 0.32358861000000005
# m = 3163 success_rate = 0.0 avg_error = 0.03382805965401979 t = 0.88315981
# m = 10001 success_rate = 0.0 avg_error = 0.028733244615907105 t = 2.87407894
# m = 31623 success_rate = 0.0 avg_error = 0.022105334676399822 t = 9.03562879
# m = 100001 success_rate = 0.01 avg_error = 0.014884901570851363 t = 28.517210979999998
# m = 316228 success_rate = 0.03 avg_error = 0.007784985268357197 t = 89.49684094
# m = 1000001 success_rate = 0.03 avg_error = 0.00437078058111167 t = 287.85801966
# m = 3162278 success_rate = 0.05 avg_error = 0.002747090669493193 t = 908.13312835


# meta_sampler(multi_gradient_descent, [trunc(Int, (ceil(4*sqrt(2)^x))) for x in 0:13], "Multi Gradient Descent Search")
# [4, 6, 9, 12, 17, 23, 33, 46, 65, 91, 129, 182, 257, 363]
# m = 4 success_rate = 0.082 avg_error = 0.03665203106790557 t = 1.035536088
# m = 6 success_rate = 0.105 avg_error = 0.030286225376235865 t = 1.531028402
# m = 9 success_rate = 0.148 avg_error = 0.0275077619865755 t = 2.261934664
# m = 12 success_rate = 0.2 avg_error = 0.02532079621873848 t = 3.009235635
# m = 17 success_rate = 0.212 avg_error = 0.02485047888919901 t = 4.217542385
# m = 23 success_rate = 0.317 avg_error = 0.02156685985839548 t = 5.708143581
# m = 33 success_rate = 0.433 avg_error = 0.017942024525335675 t = 8.272503136
# m = 46 success_rate = 0.488 avg_error = 0.01621976937107743 t = 11.162790875
# m = 65 success_rate = 0.626 avg_error = 0.01190885153779675 t = 15.590694013
# m = 91 success_rate = 0.767 avg_error = 0.007506689940931248 t = 21.623156169
# m = 129 success_rate = 0.886 avg_error = 0.0037914603580728227 t = 30.505902099
# m = 182 success_rate = 0.942 avg_error = 0.0020441682375977347 t = 42.96434712
# m = 257 success_rate = 0.985 avg_error = 0.0007024765186366006 t = 60.338082316
# m = 363 success_rate = 0.994 avg_error = 0.0004216192235201725 t = 83.335423626



function driver_plot_runtme_vs_avg_error()

    gd = [0.03665203106790557, 1.035536088,
    0.030286225376235865, 1.531028402,
    0.0275077619865755, 2.261934664,
    0.02532079621873848, 3.009235635,
    0.02485047888919901, 4.217542385,
    0.02156685985839548, 5.708143581,
    0.017942024525335675, 8.272503136,
    0.01621976937107743, 11.162790875,
    0.01190885153779675, 15.590694013,
    0.007506689940931248, 21.623156169,
    0.0037914603580728227, 30.505902099,
    0.0020441682375977347, 42.96434712,
    0.0007024765186366006, 60.338082316,
    0.0004216192235201725, 83.335423626]

    rs = [0.06937220969827483, 0.028327790000000002,
    0.05395262138616602, 0.08155611,
    0.04230030205385441, 0.32358861000000005,
    0.03382805965401979, 0.88315981,
    0.028733244615907105, 2.87407894,
    0.022105334676399822, 9.03562879,
    0.014884901570851363, 28.517210979999998,
    0.007784985268357197, 89.49684094,
    0.00437078058111167, 287.85801966,
    0.002747090669493193, 908.13312835]

    gd_x = []
    gd_y = []

    rs_x = []
    rs_y = []

    for i in 1:2:length(gd)
        push!(gd_y, gd[i])
    end

    for i in 2:2:length(gd)
        push!(gd_x, gd[i])
    end

    for i in 1:2:length(rs)
        push!(rs_y, rs[i])
    end

    for i in 2:2:length(rs)
        push!(rs_x, rs[i])
    end


    println("gd_x ", gd_x)
    println("gd_x ", gd_y)
    println("rs_x ", rs_x)
    println("rs_y ", rs_y)

    gt_success_rate = [0.082,
    0.105,
    0.148,
    0.2,
    0.212,
    0.317,
    0.433,
    0.488,
    0.626,
    0.767,
    0.886,
    0.942,
    0.985,
    0.994,]
    println("gt_success_rate ", gt_success_rate)

    line_width = 4
    title = "random sampling vs multi-gradient descent"
    PlotlyJS.plot(
        [
            PlotlyJS.scatter(x=gd_x, y=gd_y, name="multi-gradient descent", yaxis = "y1", line_color = "red", line_width = line_width),
            PlotlyJS.scatter(x=rs_x, y=rs_y, name="random sampling", yaxis = "y1", line_color = "blue", line_width = line_width)
        ],
        Layout(
            font_size=32,
            xaxis_domain=[0.15, 0.95],
            yaxis_domain=[0.0, 0.95],
            xaxis_type="log",
            yaxis=attr(
                title="average error", titlefont_color="red", position = 0.15,
            ),

            title_text=title * "<br>runtime vs average_error",
            xaxis_title = "runtime (ms)"
        )
    )

end

# driver_plot_runtme_vs_avg_error()


function driver_meta_sampler()
#     meta_sampler(sampling_search, [trunc(Int, (ceil(100*(sqrt(10)^x)))) for x in 0:9], "Sampling Search")
#     meta_sampler(multi_gradient_descent, [trunc(Int, (ceil(4*sqrt(2)^x))) for x in 0:13], "Multi Gradient Descent Search")
    meta_sampler(hybrid_search, [trunc(Int, (ceil(300*(sqrt(sqrt(10))^x)))) for x in 0:11], "Hybrid Search")
end

# driver_meta_sampler()

#=
sandbox:
- Julia version: 1.7.2
- Author: kliment
- Date: 2022-05-06
=#

# meta_sampler(hybrid_search, [trunc(Int, (ceil(300*(sqrt(sqrt(10))^x)))) for x in 0:11], "Hybrid Search")
# ms [300, 534, 949, 1688, 3000, 5335, 9487, 16871, 30000, 53349, 94869, 168703]
# meta_samples 1000
# m = 300 success_rate = 0.033 avg_error = 0.036018935563254924 t(ms) = 0.220624291
# m = 534 success_rate = 0.047 avg_error = 0.03262129535610725 t(ms) = 0.298234835
# m = 949 success_rate = 0.062 avg_error = 0.0300346437549304 t(ms) = 0.421984776
# m = 1688 success_rate = 0.11 avg_error = 0.028065086365631015 t(ms) = 0.656963664
# m = 3000 success_rate = 0.144 avg_error = 0.026968661280443545 t(ms) = 1.086018741
# m = 5335 success_rate = 0.225 avg_error = 0.024437559053308855 t(ms) = 1.839878077
# m = 9487 success_rate = 0.34 avg_error = 0.020844450227038538 t(ms) = 3.154896856
# m = 16871 success_rate = 0.508 avg_error = 0.015596989950232974 t(ms) = 5.539893427
# m = 30000 success_rate = 0.697 avg_error = 0.009694063041909616 t(ms) = 9.74070325
# m = 53349 success_rate = 0.882 avg_error = 0.003919162328555083 t(ms) = 17.229247389
# m = 94869 success_rate = 0.97 avg_error = 0.0011736983719339694 t(ms) = 30.206769983
# m = 168703 success_rate = 0.999 avg_error = 0.0002687308028545272 t(ms) = 53.575674859
# ms [300, 534, 949, 1688, 3000, 5335, 9487, 16871, 30000, 53349, 94869, 168703]
# success_rates Any[0.033, 0.047, 0.062, 0.11, 0.144, 0.225, 0.34, 0.508, 0.697, 0.882, 0.97, 0.999]
# avg_errors Any[0.036018935563254924, 0.03262129535610725, 0.0300346437549304, 0.028065086365631015, 0.026968661280443545, 0.024437559053308855, 0.020844450227038538, 0.015596989950232974, 0.009694063041909616, 0.003919162328555083, 0.0011736983719339694, 0.0002687308028545272]
# ts Any[0.220624291, 0.298234835, 0.421984776, 0.656963664, 1.086018741, 1.839878077, 3.154896856, 5.539893427, 9.74070325, 17.229247389, 30.206769983, 53.575674859]

# multi gradient descent
# ms [4, 6, 9, 12, 17, 23, 33, 46, 65, 91, 129, 182, 257, 363]
# success_rate [0.082, 0.105, 0.148, 0.2, 0.212, 0.317, 0.433, 0.488, 0.626, 0.767, 0.886, 0.942, 0.985, 0.994]
# ts [1.035536088, 1.531028402, 2.261934664, 3.009235635, 4.217542385, 5.708143581, 8.272503136, 11.162790875, 15.590694013, 21.623156169, 30.505902099, 42.96434712, 60.338082316, 83.335423626]
# avg_errors [0.03665203106790557, 0.030286225376235865, 0.0275077619865755, 0.02532079621873848, 0.02485047888919901, 0.02156685985839548, 0.017942024525335675, 0.01621976937107743, 0.01190885153779675, 0.007506689940931248, 0.0037914603580728227, 0.0020441682375977347, 0.0007024765186366006, 0.0004216192235201725]

# random sampling
# ts [0.028327790000000002, 0.08155611, 0.32358861000000005, 0.88315981, 2.87407894, 9.03562879, 28.517210979999998, 89.49684094, 287.85801966, 908.13312835]
# avg_errors [0.06937220969827483, 0.05395262138616602, 0.04230030205385441, 0.03382805965401979, 0.028733244615907105, 0.022105334676399822, 0.014884901570851363, 0.007784985268357197, 0.00437078058111167, 0.002747090669493193]



function plot_xs_vs_ys(xaxis_name, xaxis_units, yaxis_name, data_xs_s, data_ys_s, names, colors)

    @assert length(data_xs_s) == length(data_ys_s)
    @assert length(data_xs_s) == length(names)
    @assert length(data_xs_s) == length(colors)

    N = length(data_xs_s)

    for i in 1:N
        @assert length(data_xs_s[i]) == length(data_ys_s[i])
    end

    y_lines = GenericTrace[]

    line_width = 4
    for i in 1:N
        line = PlotlyJS.scatter(x=data_xs_s[i], y=data_ys_s[i], name=names[i], yaxis = "y1", line_color = colors[i], line_width = line_width)
        push!(y_lines, line)
    end

    title = names[1]
    for i in 2:N
        title = title * " vs "
        title = title * names[i]
    end

    PlotlyJS.plot(
        y_lines,
        Layout(
            font_size=32,
            xaxis_domain=[0.15, 0.95],
            yaxis_domain=[0.0, 0.95],
            xaxis_type="log",
            yaxis=attr(
                title=yaxis_name, titlefont_color="purple", position = 0.15,
            ),
            title_text=title * "<br>" * xaxis_name * " vs "*yaxis_name,
            xaxis_title = xaxis_name * " (" * xaxis_units * ")"
        )
    )

end

function driver_plot_runtime_vs_average_error()

    data_xs_s = [
        [0.220624291, 0.298234835, 0.421984776, 0.656963664, 1.086018741, 1.839878077, 3.154896856, 5.539893427, 9.74070325, 17.229247389, 30.206769983, 53.575674859],
        [1.035536088, 1.531028402, 2.261934664, 3.009235635, 4.217542385, 5.708143581, 8.272503136, 11.162790875, 15.590694013, 21.623156169, 30.505902099, 42.96434712, 60.338082316, 83.335423626],
        [0.028327790000000002, 0.08155611, 0.32358861000000005, 0.88315981, 2.87407894, 9.03562879, 28.517210979999998, 89.49684094, 287.85801966, 908.13312835]
    ]

    # average error
#     data_ys_s = [
#         [0.036018935563254924, 0.03262129535610725, 0.0300346437549304, 0.028065086365631015, 0.026968661280443545, 0.024437559053308855, 0.020844450227038538, 0.015596989950232974, 0.009694063041909616, 0.003919162328555083, 0.0011736983719339694, 0.0002687308028545272],
#         [0.03665203106790557, 0.030286225376235865, 0.0275077619865755, 0.02532079621873848, 0.02485047888919901, 0.02156685985839548, 0.017942024525335675, 0.01621976937107743, 0.01190885153779675, 0.007506689940931248, 0.0037914603580728227, 0.0020441682375977347, 0.0007024765186366006, 0.0004216192235201725],
#         [0.06937220969827483, 0.05395262138616602, 0.04230030205385441, 0.03382805965401979, 0.028733244615907105, 0.022105334676399822, 0.014884901570851363, 0.007784985268357197, 0.00437078058111167, 0.002747090669493193]
#     ]

    # success rate
    data_ys_s = [
        [0.033, 0.047, 0.062, 0.11, 0.144, 0.225, 0.34, 0.508, 0.697, 0.882, 0.97, 0.999], #2x for 0.882
        [0.082, 0.105, 0.148, 0.2, 0.212, 0.317, 0.433, 0.488, 0.626, 0.767, 0.886, 0.942, 0.985, 0.994],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.3, 0.5]
    ]

    names = [
        "hybrid search",
        "multi-gradient descent",
        "random_sampling"
    ]

    colors = [
        "green",
        "red",
        "blue"
    ]


#     N = length(data_xs_s)
#     y_lines = GenericTrace[]
#
#     line_width = 4
#     for i in 1:N
#         line = PlotlyJS.scatter(x=data_xs_s[i], y=data_ys_s[i], name=names[i], yaxis = "y1", line_color = colors[i], line_width = line_width)
#         push!(y_lines, line)
#     end
#
# #     line_width = 4
# #     y_lines = [
# #             PlotlyJS.scatter(x=data_xs_s[1], y=data_ys_s[1], name=names[1], yaxis = "y1", line_color = "green", line_width = line_width),
# #             PlotlyJS.scatter(x=data_xs_s[2], y=data_ys_s[2], name=names[2], yaxis = "y1", line_color = "blue", line_width = line_width),
# #             PlotlyJS.scatter(x=data_xs_s[3], y=data_ys_s[3], name=names[3], yaxis = "y1", line_color = "orange", line_width = line_width)
# #         ]
#
#     title = "random sampling vs multi-gradient descent"
#     xaxis_name = "runtime"
#     yaxis_name = "average_error"
#     xaxis_units = "ms"
#     PlotlyJS.plot(
#         y_lines,
#         Layout(
#             font_size=32,
#             xaxis_domain=[0.15, 0.95],
#             yaxis_domain=[0.0, 0.95],
#             xaxis_type="log",
#             yaxis=attr(
#                 title="average error", titlefont_color="red", position = 0.15,
#             ),
#
#             title_text=title * "<br>" * xaxis_name * " vs " * yaxis_name,
#             xaxis_title = xaxis_name * " (" * xaxis_units * ")"
#         )
#     )

    plot_xs_vs_ys("runtime", "ms", "success rate", data_xs_s, data_ys_s, names, colors)

end

# driver_plot_runtime_vs_average_error()