#=
main:
- Julia version: 
- Author: kliment
- Date: 2022-05-05
=#

include("../definitions.jl")

function f(p, x)
    if(x < p[1])
        if (x < p[2])
            return p[3]
        else
            return p[4]
        end
    else
        if(x < p[5])
            return p[6]
        else
            return p[7]
        end
    end
end

function harness_f(p, x, y)
    return (f(p, x) - y)^2
end

function dag_f(p, inputs) # meta-harness
    n = size(inputs, 1)
    error = 0
    for i in 1:n
        error += harness_f(p, inputs[i][1], inputs[i][2])
    end
    return error
end


# Pkg.add("ForwardDiff")
using ForwardDiff

function gradient_descent_from_one_initialization(the_dag, local_p)
    g(local_local_p) = ForwardDiff.gradient(the_dag, local_local_p)

    #zygote.gradient
    #julia slack #autodiff channel

    # Optimisers.jl

    alpha = 0.001
    delta = 1
    error = the_dag(local_p)

    while(error > 0.000000001 && delta > 0.0000000000000001)
        prev_error = error
        local_p = local_p - alpha*g(local_p)
        error = the_dag(local_p)
        delta = prev_error-error
    end

    return error, local_p
end




# println("init eval:")
# println("error: ", curried_dag_f(p))
# n = length(inputs)
# for i in 1:n
#     println(f(p, inputs[i][1]), " ", inputs[i][2]);
# end
#
# error, final_p = gradient_descent_from_one_initialization(curried_dag_f, p, inputs)
#
# println("final eval:")
# println("error: ", curried_dag_f(final_p), " returned error: ", error)
# n = length(inputs)
# for i in 1:n
#     println(f(p, inputs[i][1]), " ", inputs[i][2]);
# end

function multi_gradient_descent(num_trials, the_dag, num_params)

    all_final_errors = Array{Tuple{MyFloat, Int64}}(undef, num_trials);
    all_ps = Array{Vector{Vector{MyFloat}}}(undef, num_trials);
    Threads.@threads for i in 1:num_trials
        local_p = rand(MyFloat, num_params)
        local_error, final_p = gradient_descent_from_one_initialization(the_dag, local_p)
        all_final_errors[i] = (local_error, i)
        all_ps[i] = [final_p]
    end

    all_final_errors = sort(all_final_errors)

    final_error = all_final_errors[1][1]
    final_p_id = all_final_errors[1][2]

    return final_error, all_ps[final_p_id]

end

using Plots

function plot_how_many_samples_you_need_for_solution()
    println("Threads.nthreads() ", Threads.nthreads())

    inputs = [[0, 0], [1/4, 1/4], [2/4, 2/4], [3/4, 3/4]]
    num_params = 7

    function curried_dag_f(local_p)
        return dag_f(local_p, inputs)
    end

    println("--compile--")
    final_error, final_p = multi_gradient_descent(1, curried_dag_f, num_params)
    println("--run--")

    xs = []
    ys = []

    for num_trials in 5:5:120

        function run_meta_trials(meta_trials)
            errors = Array{MyFloat}(undef, meta_trials);
            Threads.@threads for i in 1:meta_trials
                final_error, final_p = multi_gradient_descent(num_trials, curried_dag_f, num_params)
                errors[i] = final_error < 0.000001;
            end
            println
            return sum(errors)/meta_trials
        end

        t = @elapsed avg_error = run_meta_trials(100)

        push!(xs, num_trials)
        push!(ys, avg_error)

        println("num_trials: ", num_trials, " avg error = ", avg_error, " t = ", t)

    end

    p = plot(title = "How many trials you need to find global optimum \n case study: 7 parameter decision tree.", xaxis = "number of trials", yaxis = "% success")
    plot!(p, xs, ys, label = "% chance of success")
    display(p)

    readline()
end

# plot_how_many_samples_you_need_for_solution()

"""
without paralelization
num_trials: 5 avg error = 0.032291666798801695 t = 1.265677862
num_trials: 10 avg error = 0.029166666734977576 t = 0.217848514
num_trials: 15 avg error = 0.027083333466889632 t = 0.316370243
num_trials: 20 avg error = 0.02291666692951575 t = 0.419148365
num_trials: 25 avg error = 0.018750000393785396 t = 0.531674496
num_trials: 30 avg error = 0.01875000039221981 t = 0.6399249
num_trials: 35 avg error = 0.014583333855755172 t = 0.74180371
num_trials: 40 avg error = 0.017708333758692775 t = 0.842050883
num_trials: 45 avg error = 0.01250000058609411 t = 0.943469202
num_trials: 50 avg error = 0.012500000586293603 t = 1.05490807
num_trials: 55 avg error = 0.013541667222407888 t = 1.160661078
num_trials: 60 avg error = 0.008333334052326705 t = 1.262337643
num_trials: 65 avg error = 0.009375000686228244 t = 1.368584747
num_trials: 70 avg error = 0.012500000587383915 t = 1.485171763
num_trials: 75 avg error = 0.0052083341471351905 t = 1.591742889
num_trials: 80 avg error = 0.007291667418702582 t = 1.675142617
num_trials: 85 avg error = 0.009375000682348847 t = 1.811835703
num_trials: 90 avg error = 0.010416667319085252 t = 1.961452156
num_trials: 95 avg error = 0.0031250008750344418 t = 1.946189519
num_trials: 100 avg error = 0.006250000777386446 t = 1.948827151

num_trials: 5 avg error = 0.03125000013382448 t = 1.256476802
num_trials: 10 avg error = 0.026041666830336974 t = 0.208192135
num_trials: 15 avg error = 0.025000000197769677 t = 0.307004253
num_trials: 20 avg error = 0.0177083337582292 t = 0.404361996
num_trials: 25 avg error = 0.02187500029576295 t = 0.521036348
num_trials: 30 avg error = 0.02500000019793084 t = 0.618824344
num_trials: 35 avg error = 0.02291666692822482 t = 0.708014125
num_trials: 40 avg error = 0.017708333758842523 t = 0.824195799
num_trials: 45 avg error = 0.015625000490934824 t = 0.927686243
num_trials: 50 avg error = 0.01666666712571255 t = 1.053756864
num_trials: 55 avg error = 0.00937500068458066 t = 1.133879779
num_trials: 60 avg error = 0.01250000058552908 t = 1.231951272
num_trials: 65 avg error = 0.011458333954243267 t = 1.34783701
num_trials: 70 avg error = 0.009375000681889185 t = 1.442206423
num_trials: 75 avg error = 0.014583333856313895 t = 1.554742906
num_trials: 80 avg error = 0.0020833342451290816 t = 1.661694965
num_trials: 85 avg error = 0.007291667416059609 t = 1.760582465
num_trials: 90 avg error = 0.00937500068256516 t = 1.852674584
num_trials: 95 avg error = 0.00416666751252181 t = 1.925602567
num_trials: 100 avg error = 0.008333334051785744 t = 1.969865646


After paralelization 12 cores [inner loop]

num_trials: 5 avg error = 0.030208333465559054 t = 1.281301982
num_trials: 10 avg error = 0.026041666865246612 t = 0.13622595
num_trials: 15 avg error = 0.028125000100745735 t = 0.207943312
num_trials: 20 avg error = 0.01979166702897835 t = 0.334179094
num_trials: 25 avg error = 0.02187500029670596 t = 0.372369312
num_trials: 30 avg error = 0.01979166702495843 t = 0.391479598
num_trials: 35 avg error = 0.01770833375760655 t = 0.501017728
num_trials: 40 avg error = 0.01145833395460221 t = 0.507564471
num_trials: 45 avg error = 0.014583333855628534 t = 0.618316762
num_trials: 50 avg error = 0.010416667318682465 t = 0.634225613
num_trials: 55 avg error = 0.014583333856146269 t = 0.843406201
num_trials: 60 avg error = 0.010416667320431929 t = 0.782165174
num_trials: 65 avg error = 0.010416667319799124 t = 1.006393526
num_trials: 70 avg error = 0.011458333952920157 t = 0.977830028
num_trials: 75 avg error = 0.005208334151067638 t = 1.102443761
num_trials: 80 avg error = 0.009375000685690332 t = 0.988590294
num_trials: 85 avg error = 0.012500000585599693 t = 1.280896975
num_trials: 90 avg error = 0.009375000682681937 t = 1.225556433
num_trials: 95 avg error = 0.006250000779497167 t = 1.261978948
num_trials: 100 avg error = 0.005208334148940306 t = 1.572616086
num_trials: 105 avg error = 0.010416667318039075 t = 1.56091737

num_trials: 5 avg error = 0.03437500003443824 t = 1.313043584
num_trials: 10 avg error = 0.028125000100031418 t = 0.132330025
num_trials: 15 avg error = 0.020833333661435787 t = 0.209898458
num_trials: 20 avg error = 0.021875000295400574 t = 0.311243538
num_trials: 25 avg error = 0.026041666831615594 t = 0.32654347
num_trials: 30 avg error = 0.021875000296099626 t = 0.412697937
num_trials: 35 avg error = 0.02187500029408953 t = 0.481029987
num_trials: 40 avg error = 0.020833333659913675 t = 0.583304292
num_trials: 45 avg error = 0.021875000294102276 t = 0.532369773
num_trials: 50 avg error = 0.009375000683985201 t = 0.753792595
num_trials: 55 avg error = 0.013541667219855953 t = 0.718938722
num_trials: 60 avg error = 0.012500000587400561 t = 0.812461265
num_trials: 65 avg error = 0.011458333954263064 t = 0.935149629
num_trials: 70 avg error = 0.006250000779995908 t = 1.140529302
num_trials: 75 avg error = 0.009375000683974156 t = 1.00783778
num_trials: 80 avg error = 0.009375000681203288 t = 1.205790217
num_trials: 85 avg error = 0.008333334050154798 t = 1.201836232
num_trials: 90 avg error = 0.009375000685761685 t = 1.327670138
num_trials: 95 avg error = 0.007291667416599403 t = 1.310902089
num_trials: 100 avg error = 0.002083334241475163 t = 1.484120785
num_trials: 105 avg error = 0.009375000681956817 t = 1.406400872

num_trials: 5 avg error = 0.018750000033821154 t = 1.461163754
num_trials: 10 avg error = 0.02291666699429454 t = 0.159947039
num_trials: 15 avg error = 0.021875000295580246 t = 0.232158681
num_trials: 20 avg error = 0.027083333466367664 t = 0.223285652
num_trials: 25 avg error = 0.02083333366020825 t = 0.38762971
num_trials: 30 avg error = 0.023958333563070144 t = 0.404353025
num_trials: 35 avg error = 0.014583333856956978 t = 0.443900551
num_trials: 40 avg error = 0.01875000039187103 t = 0.615849828
num_trials: 45 avg error = 0.016666667126946467 t = 0.636735652
num_trials: 50 avg error = 0.014583333856247315 t = 0.712639236
num_trials: 55 avg error = 0.014583333854258031 t = 0.767482813
num_trials: 60 avg error = 0.00937500068150509 t = 0.798941021
num_trials: 65 avg error = 0.007291667417338962 t = 0.874674448
num_trials: 70 avg error = 0.010416667316655859 t = 0.988553977
num_trials: 75 avg error = 0.008333334047608004 t = 1.037450108
num_trials: 80 avg error = 0.00937500068498202 t = 1.123656598
num_trials: 85 avg error = 0.008333334051947245 t = 1.188019794
num_trials: 90 avg error = 0.004166667515438991 t = 1.211754307
num_trials: 95 avg error = 0.006250000782031954 t = 1.259094246
num_trials: 100 avg error = 0.0052083341470824 t = 1.407016134
num_trials: 105 avg error = 0.009375000683649836 t = 1.455822916

"""

# println("best eval:")
# println("error: ", curried_dag_f(final_p), " returned error: ", final_error)
# n = length(inputs)
# for i in 1:n
#     println(f(final_p, inputs[i][1]), " ", inputs[i][2]);
# end


# println("ground truth:")
# final_p = [1/4+1/8, 1/8, 0, 1/4, 2/4+1/8, 2/4, 3/4]
# println("error: ", curried_dag_f(final_p), " should be error: ", 0)
# n = length(inputs)
# for i in 1:n
#     println(f(final_p, inputs[i][1]), " ", inputs[i][2]);
# end


