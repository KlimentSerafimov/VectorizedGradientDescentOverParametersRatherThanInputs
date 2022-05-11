#=
simple_vectorization:
- Julia version: 1.7.2
- Author: kliment
- Date: 2022-05-05
=#

function for_loop(x, y, rez)
    n = length(x)
    for i in 1:n
        rez[i] = x[i] + y[i]
    end
    return rez
end

function for_loop_simd(x, y, rez)
    n = length(x)
    @simd for i in 1:n
        rez[i] = x[i] + y[i]
    end
    return rez
end

# function my_benchmark(f, trials = 5)
#     sum_times = 0
#     min_time = 1000
#     for i in 1:trials
#         time = @elapsed f()
#         sum_times+=time
#         min_time = min(min_time, time)
#     end
#
#     println("min_time ", min_time)
#     println("avg_time ", sum_times/trials)
# end

using BenchmarkTools


function array_sum()
    n = 1000000
    the_type = MyFloat
    x = rand(MyFloat, the_type, n)
    y = rand(MyFloat, the_type, n)

    rez = zeros(MyFloat, n)
    sum_time_for_simd_before = @benchmark rez = $for_loop_simd($x,$y, $rez)

    sum_time_vect = @benchmark rez = $x .+ $y
    sum_time_non_vect = @benchmark rez = $x + $y # wont get fused with other broadcasts

    sum_time_for = @benchmark rez = $for_loop($x,$y, $rez)
    sum_time_for_simd_after = @benchmark rez = $for_loop_simd($x,$y, $rez)

    println("sum_time_vect     ", sum_time_vect)
    println("sum_time_non_vect ", sum_time_non_vect)
    println("sum_time_for      ", sum_time_for)
    println("sum_time_for_simd_before ", sum_time_for_simd_before)
    println("sum_time_for_simd_after ", sum_time_for_simd_after)

end

# function matrix_sum()
#     n = 1000000

#     the_type = MyFloat
#     x = rand(MyFloat, the_type, 1000000)
#     y = rand(MyFloat, the_type, 1000000)
#
#     sum_time_vect = @elapsed rez = x .+ y
#     sum_time_non_vect = @elapsed rez = x + y
#
#     println("sum_time_vect     ", sum_time_vect)
#     println("sum_time_non_vect ", sum_time_non_vect)
# end

println("--compile--")
array_sum()
println("--run--")
array_sum()
println("--done--")


"""

n = 1000000
the_type = MyFloat
sum_time_vect     0.004034302
sum_time_non_vect 0.004106503
sum_time_vect     0.00395139
sum_time_non_vect 0.003961708



n = 1000000
the_type = MyFloat
sum_time_vect     0.001994063
sum_time_non_vect 0.002044777
sum_time_vect     0.002066749
sum_time_non_vect 0.002028543

n = 1000000
the_type = MyFloat
sum_time_vect     0.003188043
sum_time_non_vect 0.003178072
sum_time_vect     0.003154798
sum_time_non_vect 0.003174066
"""
