#Operations specifique au pb de solveurs asymetrique sous la forme 
# MA = [0 A;
#       A' 0]

#On suppose n le paramètre de taille du probleme
#A est dans (n^2,n^2) 
#MA est dans (2n^2,2n^2)
#b est dans (n^2)
#Mxn est dans (2n^2) et est sous la forme (xn ; zeros(n^2) )

#On suppose ici les vn sous la forme(n^2+1) avec le dernier indice étant l'indice ou se situe 
# les coordonnées non nulles de vn.

using LinearAlgebra


include("StructPb.jl")

using .MyHNVs

import Base.:+ 
import Base.:*
import Base.:/
import Base.:-

function ChangeLocation(h::MyHNV)  
    if h.location == 1
        return 2
    else 
        return 1 
    end
end

function +(h1:: MyHNV, h2::MyHNV)
    #@assert h1.size == h2.size "X1 et X2 doivent etre de même taille"
    if h1.vector == zeros(h1.size)
        return h2
    elseif h2.vector == zeros(h2.size)
        return h1
    else
        #@assert h1.location == h2.location "X1 et X2 doivent être situé au même endroit pour être additionné "
        h_new = MyHNV(h1.size,h1.vector + h2.vector, h1.location )
        return h_new
    end
end 

function *(n::Number, h::MyHNV)
    h_new= MyHNV(h.size, n*h.vector, h.location)
    return h_new
end

function *(h::MyHNV, n::Number)
    return n*h
end

function /(h::MyHNV, n::Number)
    #@assert n !=0 "Division by Zero error" 
    return n^(-1)*h
end

function -(h1:: MyHNV, h2::MyHNV)
    return h1 + (-1)*h2
end

function *(A::Matrix{Float64}, h::MyHNV)
    #@assert size(A)[2] == h.size "Nb de colonnes de A et taille de x doivent etre égaux"
    #println(A*h.vector)
    if h.location == 1
        #println("location :", h.location)
        #println(h.vector)
        #println(A'*h.vector)
        h_new = MyHNV(h.size, A*h.vector, ChangeLocation(h))
    else 
        #println("location :", h.location)
        #println(h.vector)
        #println(A*h.vector)
        h_new = MyHNV(h.size, A*h.vector, ChangeLocation(h))
    end
    return h_new
end 

function *(A::Adjoint{Float64, Matrix{Float64}}, h::MyHNV)
    #@assert size(A)[2] == h.size "Nb de colonnes de A et taille de x doivent etre égaux"
    #println(A*h.vector)
    if h.location == 1
        #println("location :", h.location)
        #println(h.vector)
        #println(A'*h.vector)
        h_new = MyHNV(h.size, A*h.vector, ChangeLocation(h))
    else 
        #println("location :", h.location)
        #println(h.vector)
        #println(A*h.vector)
        h_new = MyHNV(h.size, A*h.vector, ChangeLocation(h))
    end
    return h_new
end 

function *(h::MyHNV, A::Matrix{Float64})
    #@assert size(A)[1] == h.size "Nb de colonnes de A et taille de x doivent etre égaux"
    if h.location ==1 
        h_new = MyHNV(h.size, (h.vector'*A)', ChangeLocation(h))
    else
        h_new = MyHNV(h.size, (h.vector'*A)', ChangeLocation(h))
    end
    return h_new
end 

function *(h::MyHNV, A::Adjoint{Float64, Matrix{Float64}})
    #@assert size(A)[1] == h.size "Nb de colonnes de A et taille de x doivent etre égaux"
    if h.location ==1 
        #println("h.location :", h.location)
        h_new = MyHNV(h.size, (h.vector'*A)', ChangeLocation(h))
    else
        #println("h.location :", h.location)
        h_new = MyHNV(h.size, (h.vector'*A)', ChangeLocation(h))
    end
    return h_new
end 

function *(h1::MyHNV, h2::MyHNV)
    #@assert h1.size == h2.size "Nb de colonnes de A et taille de x doivent etre égaux"
    if h1.location != h2.location 
       return 0.
    else 
        #println(h1.vector)
        #println(h2.vector)
        #println("h1*h2 :", h1.vector'*h2.vector)
        return (h1.vector)'*h2.vector
    end
end 

function inner(vi::MyHNV,A::Matrix{Float64},vj::MyHNV)
    #@assert vi.size==vj.size
    if  vi.location == vj.location
        return 0. 
    elseif vi.location == 1.
        #println("location :", vi.location)
        #println(typeof(A))
        #println(A*vj)
        #println(vi*vj)
        return vi*(A*vj)
    else 
        #println("location :", vi.location)
        #println(typeof(A'))
        #println(A'*vj)
        #println(vi*vj)
        #@assert vi.location == 2 "Vecteur non avec mauvaise localisation ".
        #println("(A'*vj):", (A'*vj))
        return vi*(A'*vj)
    end
end

function toVector(h::MyHNV)
    if h.location == 1 
        return [h.vector ; zeros(h.size)]
    else 
        return [zeros(h.size) ; h.vector]
    end 
end

function toMatrix(V::Array{MyHNV})
    l::Int64 =length(V)
    n = V[1].size
    M::Matrix{Float64} = zeros(2*n,l)
    for i in range(start=1, stop = l )
        if V[i].location == 1 
            M[1:n,i]= V[i].vector
        else 
            M[n+1:2*n,i]= V[i].vector
        end 
    end
    println("size(M) :", size(M))
    return M 
end 

function IsOrthonormalee(V,it)
    IsOrthonormal::Bool = true
    for i in range(start=1, stop =it)
        if !isapprox(V[it+1]*V[i], 0 ; atol = 1e-4) 
            println("V[it+1]*V[i] :", V[it+1]*V[i])
            IsOrthonormal = false
            break
        end
    end
    if !isapprox(V[it+1]*V[it+1], 1; atol =1e-4 )
        println("V[it+1]*V[it+1] :", V[it+1]*V[it+1])
        IsOrthonormal = false 
    end
    return IsOrthonormal 
end

function CbDeZeros(M::Matrix{Float64})
    NbDeZeros::Int64 = 0 
    M_size = size(M) #tuple
    for i in range(start=1, stop=M_size[1])
        for j in range(start=1, stop=M_size[2])
            if M[i,j] == 0. 
                NbDeZeros = NbDeZeros + 1 
            end 
        end
    end 
    return NbDeZeros
end 

function ArnoldiOptimized(A::Matrix{Float64},V::Array{MyHNV},it::Int64) #To Work on
    # inner(V[it],A,V[it]) == 0 par la nature du problème   
    if it == 1
        new_v::MyHNV = (A')*V[it] 
    else
        if V[it].location == 1 
            new_v = (A')*V[it]
        else 
            #@assert V[it].location == 2 
            new_v = A*V[it]
        end
        for i in range(start=1,stop=it - 1) #tester Hermitian Lanczos
           new_v = new_v - inner(V[i],A,V[it])*V[i]
        end
    end
    h::Float64 = LinearAlgebra.norm2(new_v.vector)
    #print("h", h, "\it")
    if !isapprox(h, 0; atol=1e-4) 
        vn::MyHNV = new_v/h
        push!(V,vn)
        #println("V[it+1] :", V[it+1])
    else 
        l = new_v.size
        z = MyHNV(l,zeros(l),ChangeLocation(V[n]))
        push!(V,z)
    end 
    return V
end

function LanczosOptimized(A::Matrix{Float64},V::Array{MyHNV},it::Int64) #To Work on
    # inner(V[it],A,V[it]) == 0 par la nature du problème   
    if it == 1
        new_v::MyHNV = (A')*V[it] 
    else
        if V[it].location == 1 
            new_v = (A')*V[it]
        else 
            #@assert V[it].location == 2 
            new_v = A*V[it]
        end
        #for i in range(start=1,stop=it - 1) #tester Hermitian Lanczos
           new_v = new_v - inner(V[it-1],A,V[it])*V[it-1]
           new_v = new_v - (new_v*V[it])*V[it]
        #end
    end
    h::Float64 = LinearAlgebra.norm2(new_v.vector)
    #print("h", h, "\it")
    if !isapprox(h, 0; atol=1e-4) 
        vn::MyHNV = new_v/h
        push!(V,vn)
    else 
        l = new_v.size
        z = MyHNV(l,zeros(l),ChangeLocation(V[n]))
        push!(V,z)
    end 
    return V
end