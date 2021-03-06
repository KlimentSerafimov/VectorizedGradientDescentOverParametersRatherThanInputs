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

function fs_v0(ps, x) #is the length of ps static?

    m = length(ps);

    mask1 = Array{Bool}(undef, m); #not a great way
    @inbounds @simd for i in 1:m
        mask1[i] = x < ps[i][1]
    end

    mask2 = Array{Bool}(undef, m);
    @inbounds @simd for i in 1:m
        mask2[i] = mask1[i] && x < ps[i][2]
    end

    mask3 = Array{Bool}(undef, m);
    @inbounds @simd for i in 1:m
        mask3[i] = !mask1[i] && x < ps[i][5]
    end

    ret = Array{MyFloat}(undef, m);
    @inbounds @simd for i in 1:m
        ret[i] = mask1[i]*(mask2[i]*ps[i][3] + (!mask2[i])*ps[i][4]) + (!mask1[i])*(mask3[i]*ps[i][6] + (!mask3[i])*ps[i][7])
    end

#     println("typeof(ret) ", typeof(ret))

    return ret;


end

function harness_fs_v0(ps, x, y)
    m = length(ps);
    rezs = fs_v0(ps, x);
    ys = Array{MyFloat}(undef, m)
    fill!(ys, y);

    diff = rezs .- ys
    square_diff = diff .* diff;
    return square_diff
end

function dag_fs_v0(ps, inputs)
    m = length(ps);
    n = length(inputs)
    error = zeros(MyFloat, m)
    for i in 1:n
        error = error .+ harness_fs_v0(ps, inputs[i][1], inputs[i][2])
    end
    return error
end