#=
vectorized_dt:
- Julia version: 1.7.2
- Author: kliment
- Date: 2022-05-11
=#

include("../definitions.jl")


function fs_v2(ps, x)

    m = length(ps[1]);
    xs = fill(x, m);

    mask1 = xs .< ps[1];
    mask2 = mask1 .&& (xs .< ps[2]);
    not_mask1 = .!mask1
    mask3 = not_mask1 .&& (xs .< ps[5]);  #try out .& vs .&&
    ret = (mask2 .* ps[3] .+ mask1 .* (.!mask2) .* ps[4]) .+ (mask3 .* ps[6] .+ not_mask1 .* (.!mask3) .* ps[7]) # use @.

    return ret;

end

function harness_fs_v2(ps, x, y)
    rezs = fs_v2(ps, x);
    ys = fill(y, length(ps[1]));

    diff = rezs .- ys
    square_diff = diff .* diff;
    return square_diff
end

function dag_fs_v2(ps, inputs) # USE A MATRIX. views;
    n = length(inputs)
    error = zeros(MyFloat, length(ps[1]))
    for i in 1:n
        error = error .+ harness_fs_v2(ps, inputs[i][1], inputs[i][2])
    end
    return error
end

using ForwardDiff

function gradient_descent_from_one_initialization!(the_dag, ps, errors, cutoff)

    function the_dag_wrapper(ps)
        return sum(the_dag(ps))
    end

    g(local_ps) = ForwardDiff.gradient(the_dag_wrapper, local_ps)

    alpha = 0.01
    delta = 1
    error = the_dag_wrapper(ps)

    actual_num_iters = 1

    while(error > cutoff && delta > cutoff)
        prev_error = error
        ps .= ps .- alpha.*g(ps)
        error = the_dag_wrapper(ps)
        delta = prev_error-error
        actual_num_iters+=1
    end

#     print(actual_" / ", " ")
    errors = the_dag(ps)
#     @assert(sum(errors) == error)

    return errors, ps, actual_num_iters
end

# using ThreadsX

function multi_gradient_descent(num_ps, the_dag, num_params, cutoff)

    ps = Array{Array{MyFloat}}(undef, num_params)

    for i in 1:num_params
        ps[i] = rand(MyFloat, num_ps);
    end

    errors = zeros(num_ps)

    errors, final_ps, num_iters = gradient_descent_from_one_initialization!(the_dag, ps, errors, cutoff)

    all_final_errors = Array{Tuple{MyFloat, Int64}}(undef, num_ps);

    for i in 1:num_ps
        all_final_errors[i] = (errors[i], i)
    end

    sort!(all_final_errors)

#     println(all_final_errors)

    final_error = all_final_errors[1][1]
    final_p_id = all_final_errors[1][2]

    return final_error, final_ps[final_p_id], num_iters

end

using Plots

function plot_how_many_samples_you_need_for_solution()
    println("Threads.nthreads() ", Threads.nthreads())

    inputs = Array{Array{MyFloat}}([[0, 0], [1/4, 1/4], [2/4, 2/4], [3/4, 3/4]])
    num_params = 7

    function curried_dag_fs_v2(local_p)
        return dag_fs_v2(local_p, inputs)
    end

    num_iters = 10000

    cutoff = 0.001

    meta_trials = 100

    println("--compile--")
    final_error, final_p = multi_gradient_descent(1, curried_dag_fs_v2, num_params, cutoff)
    println("--run--")

    xs = []
    ys = []

    for num_trials in 5:5:120

        function run_meta_trials(meta_trials)
            is_success = Array{MyFloat}(undef, meta_trials);
            num_iters = Array{Int64}(undef, meta_trials);
            for i in 1:meta_trials
                final_error, final_p, local_num_iters = multi_gradient_descent(num_trials, curried_dag_fs_v2, num_params, cutoff/10)
                is_success[i] = final_error < cutoff;
                num_iters[i] = local_num_iters
            end
            println
            return sum(is_success)/meta_trials, sum(num_iters)/meta_trials
        end

        t = @elapsed success_rate, avg_num_iters = run_meta_trials(meta_trials)

        push!(xs, num_trials)
        push!(ys, success_rate)

        println("num_trials: ", num_trials, " success_rate = ", success_rate, " avg_num_iters = ", avg_num_iters, " t = ", t)

    end

    p = plot(title = "How many trials you need to find global optimum \n case study: 7 parameter decision tree.", xaxis = "number of trials", yaxis = "% success")
    plot!(p, xs, ys, label = "% chance of success")
    display(p)

    readline()
end


plot_how_many_samples_you_need_for_solution()


"""

    num_iters = 10000

    cutoff = 0.001

    meta_trials = 100

this
num_trials: 5 success_rate = 0.05 avg_num_iters = 151.57 t = 3.13525318
num_trials: 10 success_rate = 0.18 avg_num_iters = 168.85 t = 5.072519608
num_trials: 15 success_rate = 0.25 avg_num_iters = 181.67 t = 10.643853018
num_trials: 20 success_rate = 0.23 avg_num_iters = 189.32 t = 9.280142877
num_trials: 25 success_rate = 0.3 avg_num_iters = 194.35 t = 12.894270006
num_trials: 30 success_rate = 0.33 avg_num_iters = 199.07 t = 16.854150632
num_trials: 35 success_rate = 0.44 avg_num_iters = 202.61 t = 23.03021548
num_trials: 40 success_rate = 0.52 avg_num_iters = 206.12 t = 25.019168184
num_trials: 45 success_rate = 0.49 avg_num_iters = 208.32 t = 31.283305719
num_trials: 50 success_rate = 0.58 avg_num_iters = 211.96 t = 35.532514857
num_trials: 55 success_rate = 0.56 avg_num_iters = 213.49 t = 41.314527513
num_trials: 60 success_rate = 0.6 avg_num_iters = 215.78 t = 46.893742015
num_trials: 65 success_rate = 0.66 avg_num_iters = 217.43 t = 53.383513912
num_trials: 70 success_rate = 0.63 avg_num_iters = 220.17 t = 61.450954723

original
num_trials: 5 success_rate = 0.06 avg_num_iters = 139.42 t = 0.266632876
num_trials: 10 success_rate = 0.14 avg_num_iters = 142.13 t = 0.29745621
num_trials: 15 success_rate = 0.18 avg_num_iters = 140.72 t = 0.444689485
num_trials: 20 success_rate = 0.18 avg_num_iters = 140.09 t = 0.58953565
num_trials: 25 success_rate = 0.25 avg_num_iters = 139.15 t = 0.714742869
num_trials: 30 success_rate = 0.47 avg_num_iters = 141.1 t = 0.874167528
num_trials: 35 success_rate = 0.47 avg_num_iters = 140.81 t = 1.017739702
num_trials: 40 success_rate = 0.57 avg_num_iters = 141.43 t = 1.168179195
num_trials: 45 success_rate = 0.53 avg_num_iters = 140.92 t = 1.308434558
num_trials: 50 success_rate = 0.56 avg_num_iters = 140.75 t = 1.453005546
num_trials: 55 success_rate = 0.58 avg_num_iters = 140.34 t = 1.599119749
num_trials: 60 success_rate = 0.64 avg_num_iters = 141.2 t = 1.755596402
num_trials: 65 success_rate = 0.69 avg_num_iters = 141.09 t = 1.93497865
num_trials: 70 success_rate = 0.75 avg_num_iters = 140.83 t = 2.053882291
num_trials: 75 success_rate = 0.7 avg_num_iters = 140.91 t = 2.206153595
num_trials: 80 success_rate = 0.7 avg_num_iters = 141.1 t = 2.386168904
num_trials: 85 success_rate = 0.65 avg_num_iters = 139.96 t = 2.455494353
num_trials: 90 success_rate = 0.75 avg_num_iters = 141.56 t = 2.629902761
num_trials: 95 success_rate = 0.73 avg_num_iters = 139.96 t = 2.745300188
num_trials: 100 success_rate = 0.8 avg_num_iters = 140.72 t = 2.9452651
num_trials: 105 success_rate = 0.79 avg_num_iters = 140.21 t = 3.0818552
num_trials: 110 success_rate = 0.79 avg_num_iters = 140.27 t = 3.168088568
num_trials: 115 success_rate = 0.89 avg_num_iters = 141.26 t = 3.391957794
num_trials: 120 success_rate = 0.82 avg_num_iters = 140.28 t = 3.506048731

"""