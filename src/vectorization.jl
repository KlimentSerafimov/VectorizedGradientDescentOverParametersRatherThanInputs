#=
vectorization:
- Julia version: 1.7.2
- Author: kliment
- Date: 2022-05-05
=#

function fs(ps, x)

    mask1 = Array{Bool}(undef, length(ps));
    for i in 1:length(mask1)
        mask1[i] = x < ps[i][1]
    end

    mask2 = Array{Bool}(undef, length(ps));
    for i in 1:length(mask2)
        mask2[i] = mask1[i] && x < ps[i][2]
    end

    mask3 = Array{Bool}(undef, length(ps));
    for i in 1:length(mask3)
        mask3[i] = !mask1[i] && x < ps[i][5]
    end

    ret = Array{Float64}(undef, length(ps));
    for i in 1:length(ret)
        ret[i] = mask1[i]*(mask2[i]*ps[i][3] + (!mask2[i])*ps[i][4]) + (!mask1[i])*(mask3[i]*ps[i][6] + (!mask3[i])*ps[i][7])
    end

    return ret;


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
end

function harness_fs(ps, x, y)
    rezs = fs(ps, x);
    ys = fill(y, length(ps));
#     ys = Array{Float64}(undef, length(ps));
#     for i in 1:length(ys)
#         ys[i] = y
#     end
    diff = rezs .- ys
    square_diff = diff .* diff;
    return square_diff
end

function dag_fs(ps, inputs)
    n = length(inputs)
    error = zeros(length(ps))
    for i in 1:n
        error = error .+ harness_fs(ps, inputs[i][1], inputs[i][2])
    end
    return error
end

function test(num_ps, num_params, the_dags, the_dag)
    ps = Array{Array{Float64}}(undef, num_ps)

    for i in 1:num_ps
        ps[i] = rand(num_params);
    end

    vectorized_time = @elapsed error_predicted = the_dags(ps);

    function ground_truth_calc()
        error_ground_truth = Array{Float64}(undef, num_ps);
        for i in 1:num_ps
            error_ground_truth[i] = the_dag(ps[i])
        end
        return error_ground_truth
    end

    non_vectorized_time = @elapsed error_ground_truth = ground_truth_calc();

    println(vectorized_time)
    println(non_vectorized_time)

#     println(error_predicted);
#     println(error_ground_truth);
#
#     println(error_predicted .- error_ground_truth);

    @assert (error_predicted == error_ground_truth);
end

include("main.jl")

function run_test()

    inputs = [[0, 0], [1/4, 1/4], [2/4, 2/4], [3/4, 3/4]]
    num_params = 7

    function curried_dag_fs(ps)
        return dag_fs(ps, inputs)
    end

    function curried_dag_f(p)
        return dag_f(p, inputs)
    end

    num_ps = 10000000

    test(num_ps, num_params, curried_dag_fs, curried_dag_f)

    """
    num_ps = 1000
    println(vectorized_time)
    println(non_vectorized_time)
    0.000622194
    8.4396e-5

    100000
    0.106731703
    0.009077736

    1000000
    0.680972912
    0.099349406

    10000000
    8.481107276
    0.988584691
    """

    test(num_ps, num_params, curried_dag_fs, curried_dag_f)
end

run_test();
