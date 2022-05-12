#=
hybrid_search:
- Julia version: 1.7.2
- Author: kliment
- Date: 2022-05-11
=#

include("../decision tree with array of arrays/vectorization_v3.jl")

function sampling_search(dag, num_params, num_ps, do_print)

    ps = Array{Array{MyFloat}}(undef, num_params)
    for i in 1:num_params
        ps[i] = rand(MyFloat, num_ps);
    end

    errors = dag(ps);

    final_error = errors[1]
    final_p_id = 1

    for i in 2:num_ps
        if final_error > errors[i]
            final_error = errors[i]
            final_p_id = i
        end
    end

    ret_p = Array{MyFloat}(undef, num_params)

    for i in 1:num_params
        ret_p[i] = ps[i][final_p_id]
    end

    if do_print
#         println("errors ", errors)
        println("final_error ", final_error)
    end

    return final_error, ret_p

    error_and_ids = Array{Tuple{MyFloat, Int64}}(undef, num_ps);

    sort!(error_and_ids)

#     println(all_final_errors)

    final_error = error_and_ids[1][1]
    final_p_id = error_and_ids[1][2]

    ret_p = Array{MyFloat}(undef, num_params)

    for i in 1:num_params
        ret_p[i] = ps[i][final_p_id]
    end

    return final_error, ret_p

end


include("../vectorized backrpop/non_vectorized_original.jl")

function gradient_descent_search(dag, init_p, cutoff, do_print)
    return gradient_descent_from_one_initialization(dag, init_p, cutoff, false)
end

function hybrid_search(dag_for_sampling, dag_for_gradient_descent, num_params, num_ps, cutoff, do_print)

    best_error, best_p = sampling_search(dag_for_sampling, num_params, num_ps, do_print)

#     @assert dag_for_gradient_descent(best_p) == best_error

#     return best_error, best_p

    if do_print
        println("error_after_sampling = ", best_error)
    end
    best_error, best_p = gradient_descent_search(dag_for_gradient_descent, best_p, cutoff, do_print)

    if do_print
        println("error_after_gradient = ", best_error)
    end

    return best_error, best_p

end

function harness_plot_num_samples_vs_avg_best_error()

    num_params = 7

    inputs = Array{Array{MyFloat}}([[0, 0], [1/4, 1/4], [2/4, 2/4], [3/4, 3/4]])
    function curried_dag_fs_v3(ps)
        return dag_fs_v3(ps, inputs)
    end

    meta_samples = 100

    num_ps_s = [10^x for x in 0:7]

    for num_ps in num_ps_s
        sum_errors = 0
        for meta_sample_id in 1:meta_samples
            best_error, best_ps = hybrid_search(curried_dag_fs_v3, num_params, num_ps)
            sum_errors+=best_error
        end

        bench_v3 = @benchmark best_error, best_ps = $hybrid_search($curried_dag_fs_v3, $num_params, $num_ps)

        println("m = ", num_ps, " avg_best_error = ", sum_errors/meta_samples, " bench = ", bench_v3)
    end

"""

with memory alloc INSIDE get_best_ps_by_sampling

num_params = 7

inputs = Array{Array{MyFloat}}([[0, 0], [1/4, 1/4], [2/4, 2/4], [3/4, 3/4]])
function curried_dag_fs_v3(ps)
    return dag_fs_v3(ps, inputs)
end

meta_samples = 100

num_ps_s = [10^x for x in 0:7]

m = 1 avg_best_error = 0.7361667005081509 bench = Trial(50.037 μs)
m = 10 avg_best_error = 0.177501976799302 bench = Trial(53.932 μs)
m = 100 avg_best_error = 0.07122307825421195 bench = Trial(83.264 μs)
m = 1000 avg_best_error = 0.042056629975990925 bench = Trial(389.291 μs)
m = 10000 avg_best_error = 0.029977279630516235 bench = Trial(3.645 ms)
m = 100000 avg_best_error = 0.014934639476304355 bench = Trial(40.850 ms)
m = 1000000 avg_best_error = 0.003986313876280321 bench = Trial(466.849 ms)
"""

"""
with memory alloc OUTSIDE get_best_ps_by_sampling

num_params = 7

inputs = Array{Array{MyFloat}}([[0, 0], [1/4, 1/4], [2/4, 2/4], [3/4, 3/4]])
function curried_dag_fs_v3(ps)
    return dag_fs_v3(ps, inputs)
end

meta_samples = 100

num_ps_s = [10^x for x in 0:7]
m = 1 avg_best_error = 0.684778851117903 bench = Trial(45.923 μs)
m = 10 avg_best_error = 0.16546147429913172 bench = Trial(48.502 μs)
m = 100 avg_best_error = 0.07220181473358318 bench = Trial(69.030 μs)
m = 1000 avg_best_error = 0.041749554983106424 bench = Trial(366.714 μs)
m = 10000 avg_best_error = 0.03115718175875318 bench = Trial(3.524 ms)
m = 100000 avg_best_error = 0.013756092222104782 bench = Trial(39.538 ms)
m = 1000000 avg_best_error = 0.0046841990778799 bench = Trial(442.252 ms)

"""


"""
with memory alloc OUTSIDE get_best_ps_by_sampling AND no sorting, returning any

num_params = 7

inputs = Array{Array{MyFloat}}([[0, 0], [1/4, 1/4], [2/4, 2/4], [3/4, 3/4]])
function curried_dag_fs_v3(ps)
    return dag_fs_v3(ps, inputs)
end

m = 1 avg_best_error = 0.6677015824472002 bench = Trial(43.386 μs)
m = 10 avg_best_error = 0.7323771989970972 bench = Trial(44.649 μs)
m = 100 avg_best_error = 0.697511970426141 bench = Trial(52.315 μs)
m = 1000 avg_best_error = 0.6023398865830224 bench = Trial(142.489 μs)
m = 10000 avg_best_error = 0.7188901676974129 bench = Trial(980.444 μs)
m = 100000 avg_best_error = 0.639540772119512 bench = Trial(9.910 ms)
m = 1000000 avg_best_error = 0.7303036576544408 bench = Trial(106.352 ms)

"""

"""
with memory alloc INSIDE get_best_ps_by_sampling AND no sorting, returning any

num_params = 7

inputs = Array{Array{MyFloat}}([[0, 0], [1/4, 1/4], [2/4, 2/4], [3/4, 3/4]])
function curried_dag_fs_v3(ps)
    return dag_fs_v3(ps, inputs)
end

m = 1 avg_best_error = 0.7084809404844681 bench = Trial(47.048 μs)
m = 10 avg_best_error = 0.6838440968247045 bench = Trial(49.454 μs)
m = 100 avg_best_error = 0.7472924259839359 bench = Trial(62.049 μs)
m = 1000 avg_best_error = 0.6507207202859315 bench = Trial(162.523 μs)
m = 10000 avg_best_error = 0.6434668651866496 bench = Trial(1.062 ms)
m = 100000 avg_best_error = 0.6492050009211867 bench = Trial(10.596 ms)
m = 1000000 avg_best_error = 0.6696886697253648 bench = Trial(123.657 ms)

"""


"""
with memory alloc INSIDE get_best_ps_by_sampling AND no sorting, returning best (without sorting)

num_params = 7

inputs = Array{Array{MyFloat}}([[0, 0], [1/4, 1/4], [2/4, 2/4], [3/4, 3/4]])
function curried_dag_fs_v3(ps)
    return dag_fs_v3(ps, inputs)
end

m = 1 avg_best_error = 0.7108014041636048 bench = Trial(47.627 μs)
m = 10 avg_best_error = 0.17589714288706004 bench = Trial(50.481 μs)
m = 100 avg_best_error = 0.06915119521793339 bench = Trial(67.674 μs)
m = 1000 avg_best_error = 0.04271607331188502 bench = Trial(211.774 μs)
m = 10000 avg_best_error = 0.030257694932017357 bench = Trial(1.540 ms)
m = 100000 avg_best_error = 0.013971316276666913 bench = Trial(15.291 ms)
m = 1000000 avg_best_error = 0.004224092744638863 bench = Trial(165.251 ms)
m = 10000000 avg_best_error = 0.001478995415678106 bench = Trial(2.555 s)

"""

end

# harness_plot_num_samples_vs_avg_best_error()


function harness_plot_num_samples_vs_time_to_get_0_error()

    num_params = 7

    inputs = Array{Array{MyFloat}}([[0, 0], [1/4, 1/4], [2/4, 2/4], [3/4, 3/4]])
    function curried_dag_fs_v3(ps)
        return dag_fs_v3(ps, inputs)
    end

    function curried_dag_f(local_p)
        return dag_f(local_p, inputs)
    end

    meta_trials = 100
#
#     num_ps_s = [10^x for x in 0:6]
    num_ps_s = [5000*x for x in 1:20]
#     num_ps_s = [1, 10, 20]

    cutoff = 0.001

    for num_ps in num_ps_s
#         sum_errors = 0
#         successes = 0
#         for meta_sample_id in 1:meta_samples
#             best_error, best_ps = hybrid_search(curried_dag_fs_v3, curried_dag_f, num_params, num_ps, cutoff/10, true)
#             sum_errors+=best_error < cutoff
#             successes+=cutoff
#         end
#
#         bench_v3 = @benchmark best_error, best_ps = $hybrid_search($curried_dag_fs_v3, $curried_dag_f, $num_params, $num_ps, $cutoff/10, false)
#
#         println("m = ", num_ps, " avg_best_error = ", sum_errors/meta_samples, " bench = ", bench_v3)


        function run_meta_trials(meta_trials)
            is_success = Array{MyFloat}(undef, meta_trials);
            num_iters = Array{Int64}(undef, meta_trials);
            sum_errors = 0
            for i in 1:meta_trials
                final_error, best_ps = hybrid_search(curried_dag_fs_v3, curried_dag_f, num_params, num_ps, cutoff/10, false)
                is_success[i] = final_error < cutoff;
                sum_errors+=final_error
            end
            return sum_errors / meta_trials, sum(is_success)/meta_trials
        end

        t = @elapsed avg_error, success_rate = run_meta_trials(meta_trials)

#         push!(xs, num_trials)
#         push!(ys, success_rate)

        println("m = ", num_ps, " success_rate = ", success_rate, " avg_error = ", avg_error, " t = ", t)

    end
end

harness_plot_num_samples_vs_time_to_get_0_error()

println("all done.")

"

wo gradient descent
m = 1 success_rate = 0.0 avg_error = 0.6852021913361124 t = 1.691086678
m = 10 success_rate = 0.0 avg_error = 0.1726749378713206 t = 0.009178547
m = 100 success_rate = 0.0 avg_error = 0.07147093079970704 t = 0.007273392
m = 1000 success_rate = 0.0 avg_error = 0.04204175740482677 t = 0.027585669
m = 10000 success_rate = 0.0 avg_error = 0.02791035675283517 t = 0.203779585
m = 100000 success_rate = 0.0 avg_error = 0.014474161234498972 t = 1.779048404
m = 1000000 success_rate = 0.04 avg_error = 0.004838393810185764 t = 23.872762881

with gradient descent
m = 1 success_rate = 0.01 avg_error = 0.09895116370560546 t = 1.701058145
m = 10 success_rate = 0.0 avg_error = 0.08893090087192039 t = 0.028411551
m = 100 success_rate = 0.04 avg_error = 0.04299846965862928 t = 0.037466234
m = 1000 success_rate = 0.05 avg_error = 0.03052951293892421 t = 0.048363858
m = 10000 success_rate = 0.26 avg_error = 0.02334346450256802 t = 0.18616172
m = 100000 success_rate = 0.99 avg_error = 0.0005486910593743285 t = 1.703875222
m = 1000000 success_rate = 1.0 avg_error = 0.00023798842253964158 t = 22.563922502

"

