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

    mask1 .= xs .< ps[:,1];
    mask2 .= mask1 .&& (xs .< ps[:,2]);
    not_mask1 .= .!mask1
    mask3 .= not_mask1 .&& (xs .< ps[:,5]);  #try out .& vs .&&
    ret .= (mask2 .* ps[:,3] .+ mask1 .* (.!mask2) .* ps[:,4]) .+ (mask3 .* ps[:,6] .+ not_mask1 .* (.!mask3) .* ps[:,7]) # use @.

end

function harness_fs_v3!(ps, x, y; m, ys, diff, square_diff, xs, mask1, mask2, not_mask1, mask3, ret)
    fs_v3!(ps, x; m, xs, mask1, mask2, not_mask1, mask3, ret);
    fill!(ys, y);

    diff .= ret .- ys
    square_diff .= diff .* diff;
end


function dag_fs_v3(ps, inputs) # USE A MATRIX. views;
    n = length(inputs)
    m = size(ps, 1)
    error = zeros(MyFloat, m)
    ys = zeros(MyFloat, m)
    diff = zeros(MyFloat, m)
    square_diff = zeros(MyFloat, m)
    xs = zeros(MyFloat, m)
    mask1 = falses(m)
    mask2 = falses(m)
    not_mask1 = falses(m)
    mask3 = falses(m)
    ret = zeros(MyFloat, m)
    for i in 1:n
        harness_fs_v3!(ps, inputs[i][1], inputs[i][2]; m, ys, diff, square_diff, xs, mask1, mask2, not_mask1, mask3, ret)
        error .= error .+ square_diff
    end
    return error
end


using BenchmarkTools

# function test_rep_array_of_arrays(num_ps, num_params, the_dags_v0, the_dags_v1, the_dags_v2, the_dag)
#     ps = Array{Array{MyFloat}}(undef, num_params) #THIS SHOULD BE A MATRIX
#     # Inner loops need to be accross columns; column-major order; python had row-major order.
#     # rand(MyFloat, n, m)
#
#     for i in 1:num_params
#         ps[i] = rand(MyFloat, num_ps);
#     end
#
#     ps_rotated = Array{Array{MyFloat}}(undef, num_ps)
#     for i in 1:num_ps
#         ps_rotated[i] = Array{MyFloat}(undef, num_params)
#         for j in 1:num_params
#             ps_rotated[i][j] = ps[j][i];
#         end
#     end
#
#     function ground_truth_calc()
#         error_ground_truth = zeros(MyFloat, num_ps)
#         for i in 1:num_ps
#             error_ground_truth[i] = the_dag(ps_rotated[i])
#         end
#         return error_ground_truth
#     end
#
#     println("--testing--")
#     error_ground_truth = ground_truth_calc();
#     error_predicted_v0 = the_dags_v0(ps_rotated);
#     @assert (error_predicted_v0 == error_ground_truth);
#     error_predicted_v1 = the_dags_v1(ps);
#     @assert (error_predicted_v1 == error_ground_truth);
#     error_predicted_v2 = the_dags_v2(ps);
#     @assert (error_predicted_v2 == error_ground_truth);
#     println("--tests-passed--")
#
#     println("--benchmark--")
#     bench_gt = @benchmark error_ground_truth = $ground_truth_calc();
#     println("gt ", bench_gt)
#
#     bench_v0 = @benchmark error_predicted_v0 = $the_dags_v0($ps_rotated);
#     println("v0 ", bench_v0)
#
#     bench_v1 = @benchmark error_predicted_v1 = $the_dags_v1($ps);
#     println("v1 ", bench_v1)
#
#     bench_v2 = @benchmark error_predicted_v2 = $the_dags_v2($ps);
#     println("v2 ", bench_v2)
#
#     println("--done--")
#
# end

function test_rep_matrix(num_ps, num_params, the_dags_v0, the_dags_v1, the_dags_v2, the_dags_v3, the_dag)

    _ps = rand(MyFloat, num_ps, num_params)

    ps = _ps

    function ground_truth_calc()
        error_ground_truth = zeros(MyFloat, num_ps)
        for i in 1:num_ps
            error_ground_truth[i] = the_dag(@view ps[i, :])
        end
        return error_ground_truth
    end

    println("--testing--")

    error_ground_truth = ground_truth_calc();

    error_predicted_v0 = the_dags_v0(ps);
    @assert (error_predicted_v0 == error_ground_truth);

    error_predicted_v1 = the_dags_v1(ps);
    @assert (error_predicted_v1 == error_ground_truth);

    error_predicted_v2 = the_dags_v2(ps);
    @assert (error_predicted_v2 == error_ground_truth);

    error_predicted_v3 = the_dags_v3(ps);
    @assert (error_predicted_v3 == error_ground_truth);

    println("--tests-passed--")

    #------------------------------

    println("--benchmark--")

    bench_gt = @benchmark error_ground_truth = $ground_truth_calc();
    println("gt ", bench_gt)
#     display(bench_gt)
#     println()

    bench_v0 = @benchmark error_predicted_v0 = $the_dags_v0($ps);
    println("v0 ", bench_v0)
#     display(bench_v0)
#     println()

    bench_v1 = @benchmark error_predicted_v1 = $the_dags_v1($ps);
    println("v1 ", bench_v1)
#     display(bench_v1)
#     println()

    bench_v2 = @benchmark error_predicted_v2 = $the_dags_v2($ps);
    println("v2 ", bench_v2)
#     display(bench_v2)
#     println()

    bench_v3 = @benchmark error_predicted_v3 = $the_dags_v3($ps);
    println("v3 ", bench_v3)
#     display(bench_v3)
#     println()

    println("summary")
    println("gt ", bench_gt)
    println("v0 ", bench_v0)
    println("v1 ", bench_v1)
    println("v2 ", bench_v2)
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

    function curried_dag_f(p)
        return dag_f(p, inputs)
    end

    function curried_dag_fs_v0(ps)
        return dag_fs_v0(ps, inputs)
    end

    function curried_dag_fs_v1(ps)
        return dag_fs_v1(ps, inputs)
    end

    function curried_dag_fs_v2(ps)
        return dag_fs_v2(ps, inputs)
    end

    function curried_dag_fs_v3(ps)
        return dag_fs_v3(ps, inputs)
    end


    num_ps = 1000000

    test_rep_matrix(num_ps, num_params, curried_dag_fs_v0, curried_dag_fs_v1, curried_dag_fs_v2, curried_dag_fs_v3, curried_dag_f)

    """
    original orientation
    num_ps = 1000000
    gt Trial(86.114 ms)
    v0 Trial(152.930 ms)
    v1 Trial(155.147 ms)
    v2 Trial(147.565 ms)
    """


    """
    rotated
    num_ps = 1000000
    gt Trial(87.630 ms)
    v0 Trial(159.654 ms)
    v1 Trial(190.410 ms)
    v2 Trial(183.521 ms)
    """

    """

    num_ps = 1000000
    gt Trial(88.438 ms)
    v0 Trial(152.637 ms)
    v1 Trial(148.248 ms)
    v2 Trial(142.685 ms)
    v3 Trial(138.982 ms)

    Float64
    num_ps = 1000000
    gt Trial(87.214 ms)
    v0 Trial(295.898 ms)
    v1 Trial(149.724 ms)
    v2 Trial(140.976 ms)
    v3 Trial(144.817 ms)

    Float32
    gt Trial(314.161 ms)
    v0 Trial(290.122 ms)
    v1 Trial(142.103 ms)
    v2 Trial(121.408 ms)
    v3 Trial(120.762 ms)

    Float16
    gt Trial(303.613 ms)
    v0 Trial(292.137 ms)
    v1 Trial(148.280 ms)
    v2 Trial(145.426 ms)
    v3 Trial(145.453 ms)

    """

    """
    transposed
    gt Trial(310.665 ms)
    v0 Trial(311.859 ms)
    v1 Trial(235.845 ms)
    v2 Trial(218.540 ms)
    v3 Trial(223.630 ms)

    not transposed

    gt Trial(302.747 ms)
    v0 Trial(288.725 ms)
    v1 Trial(147.555 ms)
    v2 Trial(139.330 ms)
    v3 Trial(143.135 ms)
    """

end

run_test();

function lt_prealloc_for!(x, y, rez)
    n = length(x)
    @inbounds @simd for i in 1:n
        rez[i] = x[i] < y[i]
    end
end

function lt_prealloc!(x, y, rez)
    rez .= x .< y
end

function lt(x, y)
    rez = x .< y
    return rez
end

function check_allocation()
    n = 10000000
    x = rand(n)
    y = rand(n)
    rez = rand(n)
    b = @benchmark $lt_prealloc_for!($x, $y, $rez)
    display(b)
    println()
    println()
    b = @benchmark $lt_prealloc!($x, $y, $rez)
    display(b)
    println()
    println()
    b = @benchmark out = $lt($x, $y)
    display(b)
    println()
    println()
end

# check_allocation()
