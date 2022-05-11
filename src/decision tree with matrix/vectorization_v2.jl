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

    m = size(ps, 1);
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



    mask1 = xs .< ps[:,1];
    mask2 = mask1 .&& (xs .< ps[:,2]);
    not_mask1 = .!mask1
    mask3 = not_mask1 .&& (xs .< ps[:,5]);  #try out .& vs .&&
    ret = (mask2 .* ps[:,3] .+ mask1 .* (.!mask2) .* ps[:,4]) .+ (mask3 .* ps[:,6] .+ not_mask1 .* (.!mask3) .* ps[:,7]) # use @.

    #v2:0.135478115
    #gt:0.200484127

    #v2:0.263901647
    #gt:0.11818761

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
