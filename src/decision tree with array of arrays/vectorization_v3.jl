#=
vectorization_v4:
- Julia version: 1.7.2
- Author: kliment
- Date: 2022-05-05
=#

#   implements a parallel version of
#     if(x < p[1])
#         if (x < p[2])
#             return p[3]
#         else
#             return p[4]
#         end
#     else
#         if(x < p[5])
#             return p[6]
#         else
#             return p[7]
#         end
#     end
function fs_v3!(ps, x; m, xs, mask1, mask2, not_mask1, mask3, ret)

    fill!(xs, x);

    #preallocate memory for mask1, mask2, mask3
    #use @. for applying . to all operators.

    mask1 = xs .< ps[1];
    mask2 = mask1 .&& (xs .< ps[2]);
    not_mask1 = .!mask1
    mask3 = not_mask1 .&& (xs .< ps[5]);  #try out .& vs .&&
    ret = (mask2 .* ps[3] .+ mask1 .* (.!mask2) .* ps[4]) .+ (mask3 .* ps[6] .+ not_mask1 .* (.!mask3) .* ps[7]) # use @.

end

function harness_fs_v3!(ps, x, y; m, ys, diff, square_diff, xs, mask1, mask2, not_mask1, mask3, ret)
    rezs = fs_v3!(ps, x; m, xs, mask1, mask2, not_mask1, mask3, ret);
    fill!(ys, y);

    diff = rezs .- ys
    square_diff = diff .* diff;
end

function dag_fs_v3(ps, inputs) # USE A MATRIX. views;
    n = length(inputs)
    m = length(ps[1])
    error = Array{MyFloat}(undef, m)
    fill!(error, 0)
    ys = Array{MyFloat}(undef, m)
    diff = Array{MyFloat}(undef, m)
    square_diff = Array{MyFloat}(undef, m)
    xs = Array{MyFloat}(undef, m)
    mask1 = BitVector(undef, m)
    mask2 = BitVector(undef, m)
    not_mask1 = BitVector(undef, m)
    mask3 = BitVector(undef, m)
    ret = Array{MyFloat}(undef, m)

    for i in 1:n
        error = error .+ harness_fs_v3!(ps, inputs[i][1], inputs[i][2]; m, ys, diff, square_diff, xs, mask1, mask2, not_mask1, mask3, ret)
    end

    return error
end


using BenchmarkTools

function test_rep_array_of_arrays(num_ps, num_params, the_dags_v0, the_dags_v1, the_dags_v2,the_dags_v3, the_dag)
    ps = Array{Array{MyFloat}}(undef, num_params) #THIS SHOULD BE A MATRIX
    # Inner loops need to be accross columns; column-major order; python had row-major order.
    # rand(MyFloat, MyFloat, n, m)

    for i in 1:num_params
        ps[i] = rand(MyFloat, num_ps);
    end

    ps_rotated = Array{Array{MyFloat}}(undef, num_ps)
    for i in 1:num_ps
        ps_rotated[i] = Array{MyFloat}(undef, num_params)
        for j in 1:num_params
            ps_rotated[i][j] = ps[j][i];
        end
    end

    function ground_truth_calc()
        error_ground_truth = zeros(MyFloat, num_ps)
        for i in 1:num_ps
            error_ground_truth[i] = the_dag(ps_rotated[i])
        end
        return error_ground_truth
    end

    println("--testing--")
    error_ground_truth = ground_truth_calc();

    error_predicted_v0 = the_dags_v0(ps_rotated);
    @assert (error_predicted_v0 == error_ground_truth);

    error_predicted_v1 = the_dags_v1(ps);
    @assert (error_predicted_v1 == error_ground_truth);

    error_predicted_v2 = the_dags_v2(ps);
    @assert (error_predicted_v2 == error_ground_truth);

    error_predicted_v3 = the_dags_v3(ps);

#     println(error_ground_truth)
#     println(error_predicted_v3)
#     println(error_ground_truth .- error_predicted_v3 .== 0)

    @assert (error_predicted_v3 == error_ground_truth);

    println("--tests-passed--")

    println("--benchmark--")
    bench_gt = @benchmark error_ground_truth = $ground_truth_calc();
    println("gt ", bench_gt)

    bench_v0 = @benchmark error_predicted_v0 = $the_dags_v0($ps_rotated);
    println("v0 ", bench_v0)

    bench_v1 = @benchmark error_predicted_v1 = $the_dags_v1($ps);
    println("v1 ", bench_v1)

    bench_v2 = @benchmark error_predicted_v2 = $the_dags_v2($ps);
    println("v2 ", bench_v2)

    bench_v3 = @benchmark error_predicted_v3 = $the_dags_v3($ps);
    println("v3 ", bench_v3)

    println("--done--")

end


include("main.jl")
include("vectorization_v2.jl")
include("vectorization_v1.jl")
include("vectorization_v0.jl")

function run_test()

    inputs = Array{Array{MyFloat}}([[0, 0], [1/4, 1/4], [2/4, 2/4], [3/4, 3/4]])
    num_params = 7

    function curried_dag_fs_v0(ps)
        return dag_fs_v0(ps, inputs)
    end

    function curried_dag_fs_v1(ps)
        return dag_fs_v1(ps, inputs)
    end

    function curried_dag_fs_v2(ps)
        return dag_fs_v2(ps, inputs)
    end

    function curried_dag_f(p)
        return dag_f(p, inputs)
    end

    function curried_dag_fs_v3(ps)
        return dag_fs_v3(ps, inputs)
    end

    num_ps = 1000000

    test_rep_array_of_arrays(num_ps, num_params, curried_dag_fs_v0, curried_dag_fs_v1, curried_dag_fs_v2, curried_dag_fs_v3,  curried_dag_f)

    """
    num_ps = 1000000
    gt Trial(108.938 ms)
    v0 Trial(848.328 ms)
    v1 Trial(115.610 ms)
    v2 Trial(106.516 ms)

    gt Trial(108.678 ms)
    v0 Trial(844.923 ms)
    v1 Trial(127.956 ms)
    v2 Trial(112.901 ms)

    MyFloat
    gt Trial(367.603 ms)
    v0 Trial(849.368 ms)
    v1 Trial(100.555 ms)
    v2 Trial(94.177 ms)
    v3 Trial(94.085 ms)

    """

end

run_test();