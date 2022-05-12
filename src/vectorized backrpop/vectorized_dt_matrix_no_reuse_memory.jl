#=
vectorized_dt:
- Julia version: 1.7.2
- Author: kliment
- Date: 2022-05-11
=#

include("../definitions.jl")


function fs_v2(ps, x)

    m = size(ps, 1);
    xs = fill(x, m);

    mask1 = xs .< ps[:,1];
    mask2 = mask1 .&& (xs .< ps[:,2]);
    not_mask1 = .!mask1
    mask3 = not_mask1 .&& (xs .< ps[:,5]);  #try out .& vs .&&
    ret = (mask2 .* ps[:,3] .+ mask1 .* (.!mask2) .* ps[:,4]) .+ (mask3 .* ps[:,6] .+ not_mask1 .* (.!mask3) .* ps[:,7]) # use @.

    return ret;

end

function harness_fs_v2(ps, x, y)
    rezs = fs_v2(ps, x);
    ys = fill(y, size(ps, 1));

    diff = rezs .- ys
    square_diff = diff .* diff;
    return square_diff
end

function dag_fs_v2(ps, inputs) # USE A MATRIX. views;
    n = length(inputs)
    error = zeros(MyFloat, size(ps, 1))
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

    m = size(ps, 1)

    cuttof_mult_m = cutoff*sqrt(m)

    while(error > cuttof_mult_m  && delta > cuttof_mult_m/10)
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

    ps = rand(MyFloat, num_ps, num_params)

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

lower cutoff for stagnation: cutoff / m
num_trials: 5 success_rate = 0.08 avg_num_iters = 192.18 t = 3.468220262
num_trials: 10 success_rate = 0.09 avg_num_iters = 227.68 t = 6.470830489
num_trials: 15 success_rate = 0.19 avg_num_iters = 247.63 t = 13.745444924
num_trials: 20 success_rate = 0.18 avg_num_iters = 261.43 t = 12.028764744
num_trials: 25 success_rate = 0.3 avg_num_iters = 273.82 t = 17.141526036
num_trials: 30 success_rate = 0.39 avg_num_iters = 283.54 t = 23.526093817
num_trials: 35 success_rate = 0.38 avg_num_iters = 289.55 t = 30.436859313
num_trials: 40 success_rate = 0.46 avg_num_iters = 297.37 t = 35.99656985
num_trials: 45 success_rate = 0.45 avg_num_iters = 302.57 t = 43.362279056
num_trials: 50 success_rate = 0.55 avg_num_iters = 308.81 t = 53.655396968
num_trials: 55 success_rate = 0.53 avg_num_iters = 313.74 t = 63.81868429
num_trials: 60 success_rate = 0.62 avg_num_iters = 318.3 t = 72.884860076
num_trials: 65 success_rate = 0.63 avg_num_iters = 321.3 t = 84.562315981
num_trials: 70 success_rate = 0.63 avg_num_iters = 325.56 t = 96.121047218
num_trials: 75 success_rate = 0.64 avg_num_iters = 328.28 t = 106.041180393
num_trials: 80 success_rate = 0.76 avg_num_iters = 331.36 t = 117.661101711
num_trials: 85 success_rate = 0.72 avg_num_iters = 334.2 t = 130.533460879
num_trials: 90 success_rate = 0.85 avg_num_iters = 337.46 t = 145.005305092

lower cutoff for BOTH error and stagnation: cutoff / m
num_trials: 5 success_rate = 0.08 avg_num_iters = 190.28 t = 3.222694748
num_trials: 10 success_rate = 0.11 avg_num_iters = 225.78 t = 6.714920916
num_trials: 15 success_rate = 0.25 avg_num_iters = 246.95 t = 14.198109515
num_trials: 20 success_rate = 0.22 avg_num_iters = 262.55 t = 12.402989648
num_trials: 25 success_rate = 0.34 avg_num_iters = 274.39 t = 16.958476535
num_trials: 30 success_rate = 0.39 avg_num_iters = 282.63 t = 22.588154468
num_trials: 35 success_rate = 0.33 avg_num_iters = 291.32 t = 29.263064742
num_trials: 40 success_rate = 0.52 avg_num_iters = 297.25 t = 33.233389987
num_trials: 45 success_rate = 0.49 avg_num_iters = 302.93 t = 41.600610616
num_trials: 50 success_rate = 0.58 avg_num_iters = 308.38 t = 51.299896552
num_trials: 55 success_rate = 0.6 avg_num_iters = 312.89 t = 57.494554188
num_trials: 60 success_rate = 0.53 avg_num_iters = 317.0 t = 67.575214307
num_trials: 65 success_rate = 0.57 avg_num_iters = 321.23 t = 77.844199322
num_trials: 70 success_rate = 0.56 avg_num_iters = 324.94 t = 87.724262252

lower cutoff for BOTH error and stagnation: cutoff * sqrt(m)
num_trials: 5 success_rate = 0.08 avg_num_iters = 189.34 t = 3.561956473
num_trials: 10 success_rate = 0.16 avg_num_iters = 198.46 t = 5.746333741
num_trials: 15 success_rate = 0.26 avg_num_iters = 204.14 t = 11.367139675
num_trials: 20 success_rate = 0.28 avg_num_iters = 207.41 t = 9.622937639
num_trials: 25 success_rate = 0.34 avg_num_iters = 212.12 t = 13.280879213
num_trials: 30 success_rate = 0.4 avg_num_iters = 213.45 t = 17.833959852
num_trials: 35 success_rate = 0.47 avg_num_iters = 215.21 t = 22.783643757
num_trials: 40 success_rate = 0.45 avg_num_iters = 217.43 t = 25.907947052
num_trials: 45 success_rate = 0.55 avg_num_iters = 218.74 t = 31.25488749
num_trials: 50 success_rate = 0.4 avg_num_iters = 219.93 t = 36.882238837
num_trials: 55 success_rate = 0.57 avg_num_iters = 221.32 t = 44.805360162
num_trials: 60 success_rate = 0.63 avg_num_iters = 222.23 t = 50.34812428
num_trials: 65 success_rate = 0.68 avg_num_iters = 223.54 t = 57.436715115
num_trials: 70 success_rate = 0.65 avg_num_iters = 224.57 t = 64.590213022
num_trials: 75 success_rate = 0.71 avg_num_iters = 225.62 t = 71.781756129

original
num_trials: 5 success_rate = 0.09 avg_num_iters = 141.05 t = 0.306228527
num_trials: 10 success_rate = 0.12 avg_num_iters = 140.95 t = 0.409243405
num_trials: 15 success_rate = 0.21 avg_num_iters = 142.23 t = 0.535679511
num_trials: 20 success_rate = 0.22 avg_num_iters = 141.15 t = 0.682164244
num_trials: 25 success_rate = 0.34 avg_num_iters = 142.15 t = 0.860312837
num_trials: 30 success_rate = 0.38 avg_num_iters = 139.44 t = 1.018884679
num_trials: 35 success_rate = 0.52 avg_num_iters = 141.47 t = 1.199555812
num_trials: 40 success_rate = 0.44 avg_num_iters = 140.93 t = 1.360839903
num_trials: 45 success_rate = 0.48 avg_num_iters = 140.88 t = 1.530177504
num_trials: 50 success_rate = 0.59 avg_num_iters = 140.5 t = 1.696068466
num_trials: 55 success_rate = 0.5 avg_num_iters = 140.74 t = 1.867085961
num_trials: 60 success_rate = 0.62 avg_num_iters = 140.97 t = 2.038447664
num_trials: 65 success_rate = 0.6 avg_num_iters = 139.47 t = 2.182147657
num_trials: 70 success_rate = 0.64 avg_num_iters = 140.37 t = 2.365557215
num_trials: 75 success_rate = 0.72 avg_num_iters = 140.53 t = 2.557458663
num_trials: 80 success_rate = 0.79 avg_num_iters = 141.14 t = 2.724011619
num_trials: 85 success_rate = 0.74 avg_num_iters = 140.52 t = 3.001147848
num_trials: 90 success_rate = 0.65 avg_num_iters = 139.69 t = 3.032932874
num_trials: 95 success_rate = 0.76 avg_num_iters = 140.95 t = 3.214893357
num_trials: 100 success_rate = 0.8 avg_num_iters = 140.62 t = 3.40580545
num_trials: 105 success_rate = 0.86 avg_num_iters = 140.67 t = 3.569247123
num_trials: 110 success_rate = 0.81 avg_num_iters = 141.0 t = 3.748029949
num_trials: 115 success_rate = 0.85 avg_num_iters = 140.27 t = 3.895185098
num_trials: 120 success_rate = 0.87 avg_num_iters = 140.08 t = 4.063295292

num_trials: 5 success_rate = 0.09 avg_num_iters = 141.05 t = 0.306228527
num_trials: 10 success_rate = 0.12 avg_num_iters = 140.95 t = 0.409243405
num_trials: 15 success_rate = 0.21 avg_num_iters = 142.23 t = 0.535679511
num_trials: 20 success_rate = 0.22 avg_num_iters = 141.15 t = 0.682164244
num_trials: 25 success_rate = 0.34 avg_num_iters = 142.15 t = 0.860312837
num_trials: 30 success_rate = 0.38 avg_num_iters = 139.44 t = 1.018884679
num_trials: 35 success_rate = 0.52 avg_num_iters = 141.47 t = 1.199555812
num_trials: 40 success_rate = 0.44 avg_num_iters = 140.93 t = 1.360839903
num_trials: 45 success_rate = 0.48 avg_num_iters = 140.88 t = 1.530177504
num_trials: 50 success_rate = 0.59 avg_num_iters = 140.5 t = 1.696068466
num_trials: 55 success_rate = 0.5 avg_num_iters = 140.74 t = 1.867085961
num_trials: 60 success_rate = 0.62 avg_num_iters = 140.97 t = 2.038447664
num_trials: 65 success_rate = 0.6 avg_num_iters = 139.47 t = 2.182147657
num_trials: 70 success_rate = 0.64 avg_num_iters = 140.37 t = 2.365557215
num_trials: 75 success_rate = 0.72 avg_num_iters = 140.53 t = 2.557458663
num_trials: 80 success_rate = 0.79 avg_num_iters = 141.14 t = 2.724011619
num_trials: 85 success_rate = 0.74 avg_num_iters = 140.52 t = 3.001147848
num_trials: 90 success_rate = 0.65 avg_num_iters = 139.69 t = 3.032932874
num_trials: 95 success_rate = 0.76 avg_num_iters = 140.95 t = 3.214893357
num_trials: 100 success_rate = 0.8 avg_num_iters = 140.62 t = 3.40580545
num_trials: 105 success_rate = 0.86 avg_num_iters = 140.67 t = 3.569247123
num_trials: 110 success_rate = 0.81 avg_num_iters = 141.0 t = 3.748029949
num_trials: 115 success_rate = 0.85 avg_num_iters = 140.27 t = 3.895185098
num_trials: 120 success_rate = 0.87 avg_num_iters = 140.08 t = 4.063295292

"""