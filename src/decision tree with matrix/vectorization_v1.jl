#=
vectorization_v1:
- Julia version: 1.7.2
- Author: kliment
- Date: 2022-05-05
=#

#=
vectorization_v0:
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

function fs_v1(ps, x)

    m = size(ps, 1);

    xs = fill(x, m);

    mask1 = xs .< ps[:, 1];
    mask2 = mask1 .&& (xs .< ps[:, 2]);
    not_mask1 = .!mask1
    mask3 = not_mask1 .&& (xs .< ps[:, 5]);
    ret = mask1 .* (mask2 .* ps[:, 3] .+ (.!mask2) .* ps[:, 4]) + not_mask1 .* (mask3 .* ps[:, 6] .+ (.!mask3) .* ps[:, 7])

    return ret;

end

function harness_fs_v1(ps, x, y)
    rezs = fs_v1(ps, x);
    ys = fill(y, size(ps, 1));

    diff = rezs .- ys
    square_diff = diff .* diff;
    return square_diff
end

function dag_fs_v1(ps, inputs)
    n = length(inputs)
    error = zeros(MyFloat, size(ps, 1))
    for i in 1:n
        error = error .+ harness_fs_v1(ps, inputs[i][1], inputs[i][2])
    end
    return error
end





#testing code





function test(num_ps, num_params, the_dags_v0, the_dags_v1, the_dag)
    ps = Array{Array{MyFloat}}(undef, num_params)

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

    vectorized_time_v0 = @elapsed error_predicted_v0 = the_dags_v0(ps_rotated);

    vectorized_time_v1 = @elapsed error_predicted_v1 = the_dags_v1(ps);

    function ground_truth_calc()
        error_ground_truth = Array{MyFloat}(undef, num_ps);
        for i in 1:num_ps
            error_ground_truth[i] = the_dag(ps_rotated[i])
        end
        return error_ground_truth
    end

    non_vectorized_time = @elapsed error_ground_truth = ground_truth_calc();

    println("v0:", vectorized_time_v0)
    println("v1:", vectorized_time_v1)
    println("gt:", non_vectorized_time)

#     println(error_predicted);
#     println(error_ground_truth);
#
#     println(error_predicted .- error_ground_truth);

    @assert (error_predicted_v0 == error_ground_truth);
    @assert (error_predicted_v1 == error_ground_truth);

end

include("main.jl")
include("vectorization_v0.jl")

function run_test()

    inputs = [[0, 0], [1/4, 1/4], [2/4, 2/4], [3/4, 3/4]]
    num_params = 7

    function curried_dag_fs_v0(ps)
        return dag_fs_v0(ps, inputs)
    end

    function curried_dag_fs_v1(p)
        return dag_fs_v1(p, inputs)
    end

    function curried_dag_f(p)
        return dag_f(p, inputs)
    end

    num_ps = 1000000

    println("--compile--")

    test(num_ps, num_params, curried_dag_fs_v0, curried_dag_fs_v1, curried_dag_f)

    println("--run--")

    test(num_ps, num_params, curried_dag_fs_v0, curried_dag_fs_v1, curried_dag_f)

    println("--done--")

    """

    num_ps = 10000000
    v0:11.329418917
    v1:2.36257865
    gt:1.342044169

    1000000
    v0:1.162600065
    v1:0.176850325
    gt:0.102394326

    """

    println()

end

# run_test();


