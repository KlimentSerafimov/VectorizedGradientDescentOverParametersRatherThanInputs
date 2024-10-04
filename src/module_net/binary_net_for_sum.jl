function dt(p, x)
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

dt_loss(p, x, y) = (dt(p, x) - y)^2

function dt_harness(p, data)
    n = length(data)
    error = 0
    for i in 1:n
        error += dt_loss(p, data[i][1], data[i][2])
    end
    return error
end

using ForwardDiff
function gradient_descent(f, p)
    dfdp(_p) = ForwardDiff.gradient(f, _p)
    cutoff = 0.00001
    alpha = 0.008
    delta = 1
    error = f(p)
    while(error > cutoff && delta > cutoff/10)
        prev_error = error
        p = p - alpha*dfdp(p)
        error = f(p)
        delta = prev_error-error
        println("error ", error, " delta ", delta)
    end
    return error, p
end

function driver_gradient_descent()

    data = [[0, 0], [1/4, 1/4], [2/4, 2/4], [3/4, 3/4]]
    dt_wrapper(p) = dt_harness(p, data)

    n = 7
    MyFloat = Float64
    init_p = rand(MyFloat, n)

    final_error, final_p = gradient_descent(dt_wrapper, init_p)
    println("final_error ", final_error)
    println("final_p ", final_p)
end

driver_gradient_descent()