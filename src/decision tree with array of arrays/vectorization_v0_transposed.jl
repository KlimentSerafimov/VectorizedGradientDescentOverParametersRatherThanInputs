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

function fs_v0_transposed(ps, x) #is the length of ps static?

    m = length(ps[1])

    mask1 = Array{Bool}(undef, m); #not a great way
    @inbounds @simd for i in 1:m
        mask1[i] = x < ps[1][i]
    end

    mask2 = Array{Bool}(undef, m);
    @inbounds @simd for i in 1:m
        mask2[i] = mask1[i] && x < ps[2][i]
    end

    mask3 = Array{Bool}(undef, m);
    @inbounds @simd for i in 1:m
        mask3[i] = !mask1[i] && x < ps[5][i]
    end

    ret = Array{MyFloat}(undef, m);
    @inbounds @simd for i in 1:m
        ret[i] = mask1[i]*(mask2[i]*ps[3][i] + (!mask2[i])*ps[4][i]) + (!mask1[i])*(mask3[i]*ps[6][i] + (!mask3[i])*ps[7][i])
    end

    return ret;


end

function harness_fs_v0_transposed(ps, x, y)
    rezs = fs_v0_transposed(ps, x);
    ys = Array{MyFloat}(undef, length(ps[1]))
    fill!(ys, y);

    diff = rezs .- ys
    square_diff = diff .* diff;
    return square_diff
end

function dag_fs_v0_transposed(ps, inputs)
    n = length(inputs)
    error = zeros(MyFloat, length(ps[1]))
    for i in 1:n
        error = error .+ harness_fs_v0_transposed(ps, inputs[i][1], inputs[i][2])
    end
#     println("typeof(error) ", typeof(error))
    return error
end