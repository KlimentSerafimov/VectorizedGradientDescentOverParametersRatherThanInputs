#=
simple_sum:
- Julia version: 1.7.2
- Author: kliment
- Date: 2022-05-06
=#


function my_sum!(x, ret, at)
    for i in 1:size(x, 1)
        ret[at] += x[i]
    end
end

function multi_sum!(xs, ret)
    m = size(xs, 1)
    @assert m == size(ret, 1)

    for i in 1:size(xs, 1)
        col = @view xs[i, :]
        my_sum!(col, ret, i)
    end
end

function multi_sum_transposed!(xs, ret)
    m = size(xs, 2)
    @assert m == size(ret, 1)

    for i in 1:size(xs, 2)
        col = @view xs[:, i]
        my_sum!(col, ret, i)
    end
end

using BenchmarkTools

function test__multi_sum()
    n = 100
    m = 100000
    xs = rand(MyFloat, m, n)

    ret = zeros(MyFloat, m)

    benchmark = @benchmark $multi_sum!($xs, $ret)
    display(benchmark)

#     print(ret)
end

function test__multi_sum_transposed()
    n = 100
    m = 100000
    xs = rand(MyFloat, n, m)

    ret = zeros(MyFloat, m)

    benchmark = @benchmark $multi_sum_transposed!($xs, $ret)
    display(benchmark)

#     print(ret)
end

test__multi_sum()
test__multi_sum_transposed()