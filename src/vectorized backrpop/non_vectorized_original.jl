#=
vectorized_dt:
- Julia version: 1.7.2
- Author: kliment
- Date: 2022-05-11
=#

include("../definitions.jl")
function f(p, x)
    @inbounds if(x < p[1])
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

function harness_f(p, x, y)
    return (f(p, x) - y)^2
end

function dag_f(p, inputs) # meta-harness
    n = length(inputs)
    error = 0
    for i in 1:n
        error += harness_f(p, inputs[i][1], inputs[i][2])
    end
    return error
end


using ForwardDiff
function gradient_descent_from_one_initialization(the_dag, local_p, cutoff, do_print)
    g(local_local_p) = ForwardDiff.gradient(the_dag, local_local_p)

    #zygote.gradient
    #julia slack #autodiff channel

    # Optimisers.jl

    alpha = 0.01
    delta = 1
    error = the_dag(local_p)

    num_iters = 0

    while(error > cutoff && delta > cutoff/10)
        prev_error = error
        local_p = local_p - alpha*g(local_p)
        error = the_dag(local_p)
        delta = prev_error-error
        num_iters+=1
        if do_print
            println("#", num_iters, " error ", error, " delta ", delta)
        end
    end

#     println()

    return error, local_p, num_iters
end


function multi_gradient_descent(num_ps, the_dag, num_params, cutoff)

    errors = Array{MyFloat}(undef, num_ps);
    all_ps = Array{Vector{Vector{MyFloat}}}(undef, num_ps);
    num_iters = Array{Int64}(undef, num_ps);
#     Threads.@threads
    for i in 1:num_ps
        local_p = rand(MyFloat, num_params)
        local_error, final_p, local_num_iters = gradient_descent_from_one_initialization(the_dag, local_p, cutoff)
        errors[i] = local_error
        all_ps[i] = [final_p]
        num_iters[i] = local_num_iters
    end

    ret_p = Array{MyFloat}(undef, num_params)

    final_error = errors[1]
    final_p_id = 1

    for i in 2:num_ps
        if final_error > errors[i]
            final_error = errors[i]
            final_p_id = 1
        end
    end

    return final_error, all_ps[final_p_id], floor(sum(num_iters)/num_ps)
#
#     all_final_errors = sort(all_final_errors)
#
#     final_error = all_final_errors[1][1]
#     final_p_id = all_final_errors[1][2]
#
#     return final_error, all_ps[final_p_id], floor(sum(num_iters)/num_trials)

end

using Plots

function plot_how_many_samples_you_need_for_solution()
    println("Threads.nthreads() ", Threads.nthreads())

    inputs = Array{Array{MyFloat}}([[0, 0], [1/4, 1/4], [2/4, 2/4], [3/4, 3/4]])
    num_params = 7
    function curried_dag_f(local_p)
        return dag_f(local_p, inputs)
    end

    cutoff = 0.001

    meta_trials = 100

    println("--compile--")
    final_error, final_p = multi_gradient_descent(1, curried_dag_f, num_params, cutoff)
    println("--run--")

    xs = []
    ys = []


    for num_trials in 5:5:120

        function run_meta_trials(meta_trials)
            is_success = Array{MyFloat}(undef, meta_trials);
            num_iters = Array{Int64}(undef, meta_trials);
            for i in 1:meta_trials
                final_error, final_p, local_num_iters = multi_gradient_descent(num_trials, curried_dag_f, num_params, cutoff/10)
                is_success[i] = final_error < cutoff;
                num_iters[i] = local_num_iters
            end
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


# plot_how_many_samples_you_need_for_solution()