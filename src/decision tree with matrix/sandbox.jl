#=
sandbox:
- Julia version: 1.7.2
- Author: kliment
- Date: 2022-05-06
=#

function f(p, x)
    @inbounds if(x < p[1]) #unsafe
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

rez = f(rand(MyFloat, 7), 1/2)

println(rez)


