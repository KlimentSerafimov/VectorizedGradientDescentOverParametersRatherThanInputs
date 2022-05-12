#=
vectorization_v4:
- Julia version: 1.7.2
- Author: kliment
- Date: 2022-05-05
=#

#=
vectorization_v2:
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

function fs_v2(ps, x)

    m = length(ps[1]);
    xs = fill(x, m);

    #preallocate memory for mask1, mask2, mask3
    #use @. for applying . to all operators.
    #write it as: catch

    # zygote and enzyme for gradient descent; optimized reverse-mode ad.
    # if code is already in vectorized form, they will use this
    # avx512 compilation pipeline; takes advantage of

    # Exploring broadcasting and vectorization
    # making sure it vectorizes correctly
    # takes advantage of gpu; multithredding using Threads.threads; (for future work) or distributed;

    # how do I see my code on the llvm level @code_llvm @code_native

    # take FEM code and make it run faster vs writing a generalized vectorizer

    # broadcast(f, x, y) to vectorize

    # eachcol

    # f(..) : Float
    # f.(eachol(ps), x) : Vector{Float}
    # f.(eachol(ps), x) # ps is a matrix
    # == f(ps[i], x) for each i

    #Armando's meeting.
    #SKETCH PRETTY PRINTER void BooleanDAG::mrprint(ostream& out)
    #result = runOptims(result); IS NOT CALLED thats why  --debug-output-dag doesnt work.



    mask1 = xs .< ps[1];
    mask2 = mask1 .&& (xs .< ps[2]);
    not_mask1 = .!mask1
    mask3 = not_mask1 .&& (xs .< ps[5]);  #try out .& vs .&&
    ret = (mask2 .* ps[3] .+ mask1 .* (.!mask2) .* ps[4]) .+ (mask3 .* ps[6] .+ not_mask1 .* (.!mask3) .* ps[7]) # use @.

    #v2:0.135478115
    #gt:0.200484127

    #v2:0.263901647
    #gt:0.11818761

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

using BenchmarkTools

function test(num_ps, num_params, the_dags_v0, the_dags_v2, the_dag)
    ps = Array{Array{MyFloat}}(undef, num_params) #THIS SHOULD BE A MATRIX
    # Inner loops need to be accross columns; column-major order; python had row-major order.
    # rand(MyFloat, n, m)

    for i in 1:num_params
        ps[i] = rand(MyFloat, num_ps);
    end

#     ps[i] # this is a row? FOR A MATRIX i in an index in the concatinated array;
    # if array{array} ps[i] and ps[i+1] might be in completelly different locations in memory
    # for a matrix ps[i, :] @view ps[i, :]

#     ps_rotated = transpose(ps);

    ps_rotated = Array{Array{MyFloat}}(undef, num_ps)
    for i in 1:num_ps
        ps_rotated[i] = Array{MyFloat}(undef, num_params)
        for j in 1:num_params
            ps_rotated[i][j] = ps[j][i];
        end
    end

#     println(ps[1][2], " ", ps[2][1])
#     println(ps_rotated[1][2], " ", ps_rotated[2][1])
#     println(transpose(ps)[1][2], " ", transpose(ps)[2][1])
#     println(transpose(ps_rotated)[1][2], " ", transpose(ps_rotated)[2][1])

    function ground_truth_calc()
        error_ground_truth = zeros(MyFloat, num_ps)
        for i in 1:num_ps
            error_ground_truth[i] = the_dag(ps_rotated[i])
        end
        return error_ground_truth
    end

    non_vectorized_time = @elapsed error_ground_truth = ground_truth_calc();

    vectorized_time_v2 = @elapsed error_predicted_v2 = the_dags_v2(ps);

#     vectorized_time_v0 = @elapsed error_predicted_v0 = the_dags_v0(ps_rotated);


#     println("v0:", vectorized_time_v0)
    println("v2:", vectorized_time_v2)
    println("gt:", non_vectorized_time)

#     println(error_predicted_v0);
#     println(error_predicted_v2);
#     println(error_ground_truth);

#     @assert (error_predicted_v0 == error_ground_truth);
#     @assert (error_predicted_v2 == error_ground_truth);

#     benchmark_obj = @benchmark error_predicted_v2 = $the_dags_v2($ps); # $ what is this? todo with benchamrking; tells to look globally
#
#     println(benchmark_obj)
#     display(benchmark_obj)

end

include("main.jl")
include("vectorization_v0.jl")

function run_test()

    inputs = [[0, 0], [1/4, 1/4], [2/4, 2/4], [3/4, 3/4]]
    num_params = 7

    function curried_dag_fs_v0(ps)
        return dag_fs_v0(ps, inputs)
    end

    function curried_dag_fs_v2(p)
        return dag_fs_v2(p, inputs)
    end

    function curried_dag_f(p)
        return dag_f(p, inputs)
    end

    num_ps = 1000000

    println("--compile--")

    test(num_ps, num_params, curried_dag_fs_v0, curried_dag_fs_v2, curried_dag_f)

    println("--run--")

    test(num_ps, num_params, curried_dag_fs_v0, curried_dag_fs_v2, curried_dag_f)

    println("--done--")

    """

    num_ps = 10000000
    v0:11.329418917
    v1:2.36257865
    gt:1.342044169

    v0:11.074886111
    v2:1.722353708
    gt:1.392523107

    1000000
    v0:1.162600065
    v1:0.176850325
    gt:0.102394326

    v0:0.948495581
    v2:0.122367388
    gt:0.131461056

    Float16
    v0:0.95306956
    v2:0.126210701
    gt:0.122924736

    Float16
    v0:1.190326056
    v2:0.188117966
    gt:0.1025504

    v0:1.210762586
    v2:0.173647761
    gt:0.102180644

    MyFloat
    v0:0.965848491
    v2:0.149882381
    gt:0.105865011


    multirun
    v2:0.310822187
    v2:0.128984221
    v2:0.118927847
    v2:0.12061269
    v2:0.124064999
    v2:0.135336593

    gt:0.139133488
    gt:0.130889037
    gt:0.148983374
    gt:0.122155658
    gt:0.131092369

gt:7.19e-7
gt:5.61e-7
gt:4.4e-7
gt:4.49e-7
gt:4.8e-7

    """

    println()

end

# run_test();