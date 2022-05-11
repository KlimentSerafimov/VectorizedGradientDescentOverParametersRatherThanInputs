#=
vectorization:
- Julia version: 1.7.2
- Author: kliment
- Date: 2022-05-05
=#



using BenchmarkTools

function test(num_ps, num_params, the_dags, the_dag)
    ps = Array{Array{MyFloat}}(undef, num_ps)

    for i in 1:num_ps
        ps[i] = rand(MyFloat, num_params);
    end

    vectorized_time = @elapsed error_predicted = the_dags(ps);

    function ground_truth_calc()
        error_ground_truth = Array{MyFloat}(undef, num_ps);
        for i in 1:num_ps
            error_ground_truth[i] = the_dag(ps[i])
        end
        return error_ground_truth
    end

    non_vectorized_time = @elapsed error_ground_truth = ground_truth_calc();
#     @benchmark error_ground_truth = ground_truth_calc();

    println(vectorized_time)
    println(non_vectorized_time)

#     println(error_predicted);
#     println(error_ground_truth);
#
#     println(error_predicted .- error_ground_truth);

    @assert (error_predicted == error_ground_truth);
end

include("main.jl")
include("vectorization_v0.jl")

function run_test()

    inputs = [[0, 0], [1/4, 1/4], [2/4, 2/4], [3/4, 3/4]]
    num_params = 7

    function curried_dag_fs_v0(ps)
        return dag_fs_v0(ps, inputs)
    end

    function curried_dag_f(p)
        return dag_f(p, inputs)
    end

    num_ps = 100000

    test(num_ps, num_params, curried_dag_fs_v0, curried_dag_f)

    """
    num_ps = 1000
    println(vectorized_time)
    println(non_vectorized_time)
    0.000622194
    8.4396e-5

    100000
    0.106731703
    0.009077736

    0.097760892
    0.012988438

    1000000
    0.680972912
    0.099349406

    10000000
    8.481107276
    0.988584691

    11.268527017
    1.290218418
    """

    test(num_ps, num_params, curried_dag_fs_v0, curried_dag_f)
end

run_test();


