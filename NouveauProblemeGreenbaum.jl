using LinearAlgebra


function NouveauProblemeGreenbaum(A,P,x₀, Mb, tol)
    it::Int64 = 1
    xₙ::Vector{Float64} = x₀
    d ::Int64 = length(x₀)
    Mr₀::Vector{Float64} = Mb - NewMatVectProd(A,P,xₙ) 
    normMr₀::Float64 = LinearAlgebra.norm2(Mr₀)
    normMr::Vector{Float64} = zeros(d); normMr[1] = normMr₀

    v1::Vector{Float64} = Mr₀/normMr₀
    #println("size(v1) :", size(v1))
    #V = Vector{Float64}[]; push!(V,v1)
    V = zeros(d,2) ; V[:,1]=v1
    #println("size(V[1]) :",size(V[1]))

    C = zeros(d)
    S = zeros(d)
    u = zeros(d)
    
    T_diag = zeros(d); T_upper_diag = zeros(d);
    

    R = zeros(d,2)

    ξ::Vector{Float64} = zeros(length(x₀)); ξ[1]=1.0


    while  it< length(xₙ) && normMr[it] >tol
        #println("it :", it)
        V = NewHermitianLanczos(V,A,P,it)
        println("V[:,2]'*v1 :", V[:,2]'*v1)
        @assert IsOrthonormale(V) "V doit être orthonormale"

        T_diag , T_upper_diag = NewUpdateT(T_diag , T_upper_diag,V,A,P,it)

        #println("size(T) :", size(T))

        if it > 2
            α::Float64 = S[it-2]*T_upper_diag[it-1]  
            β::Float64 = C[it-2]*T_upper_diag[it-1]
            
            η::Float64 =  (-1)*S[it-1]*β + C[it-1]*T_diag[it]
            #println("β :", β)
            #println("η :", η)
            #println("-C[it-2]* T[2][it-1] :", -C[it-2]* T[2][it-1] )
            β = C[it-1]*β  + S[it-1]*T_diag[it]

            γ::Float64 = sqrt(η^2 + T_upper_diag[it]^2)
            #println("γ :", γ)
            C[it] = η/γ
            S[it] =T_upper_diag[it]/γ
            #println("-C2T(3,4)/ sqrt(-c2(T(3,4)^2 + T(5,4)^2)) :", -C[it-2]*T[2][it-1]/sqrt((C[it-2]*T[2][it-1])^2 + T[2][it]^2))
            η = C[it]*η + S[it]*T_upper_diag[it]
            #if it%2==0
                #println("C[it] :", C[it])
                #println("S[it] :", S[it])
                #println("S[it-2]*T[2][it-1] :", S[it-2]*T[2][it-1])
                #println("α :", α)
                #println("β :", β)
                #println("η :", η)
            #end

            ρ::Vector{Float64} = (V[:,1] - β*R[:,2] - α*R[:,1])/η #V[it]
            #println("ρ :", ρ)
            R[:,1] = R[:,2]
            R[:,2] = ρ

        elseif it == 2 
            #println("C[it-1] :", C[it-1])
            #println("T[2][it-1] :", T[2][it-1])
            #println("S[it-1] :", S[it-1])
            #println("T[1][it] :", T[1][it])
            β = C[it-1]*T_upper_diag[it-1] + S[it-1]*T_diag[it]
            η =  -S[it-1]*T_upper_diag[it-1] + C[it-1]*T_diag[it]

            #println("β :", β)
            #println("η :", η)

            γ = sqrt(η^2 + T_upper_diag[it]^2)

            #println("-T(1,2)/sqrt(T(1,2)^2 + T(3,2)^2) :", -T[2][it-1]/sqrt((T[2][it-1])^2 + T[2][it]^2))
            C[it] = η/γ
            S[it] =T_upper_diag[it]/γ
            #println("C[it] :", C[it])

            η = C[it]*η + S[it]*T_upper_diag[it]
            #println("η :", η)

            ρ = (V[:,1] - β*R[:,1])/η #V[2]
            #println("ρ :", ρ)
            R[:,2] = ρ
        
        else
            γ = sqrt(T_diag[it]^2 + T_upper_diag[it]^2)
            #println("γ :", γ)
            #println()
            C[it] = T_diag[it]/γ
            S[it] =T_upper_diag[it]/γ

            η = C[it]*T_diag[it] + S[it]*T_upper_diag[it]
            #println("η :", η)

            ρ = (V[:,1])/η #V[2]
            #println("ρ :",ρ)
            R[:,1] = ρ

        end 
        
        ξ[it+1] = -S[it]*ξ[it]
        ξ[it] = C[it]*ξ[it]
        #println("ξ[it] :", ξ[it])
        #println("ξ[it+1] :", ξ[it+1])
        
        xₙ = xₙ + normMr₀*ξ[it]*ρ
        #println("xₙ :", xₙ)

        normMr[it+1] = abs(normMr₀*ξ[it+1])

        #println("NormMr :", normMr[it+1])
        
        #println("Erreur réelle :",LinearAlgebra.norm2(NewMatVectProd(A,P,xₙ) - Mb) )

        
        #if (it%20 ==0)
            #return NouveauProblemeGreenbaum(A,P,xₙ[1:Int(d/2)],xₙ[Int(d/2)+1:d], b, tol)
        #end

        it = it + 1
    end 
    #println("A*xₙ :", NewMatVectProd(A,P,xₙ) )
    return (it, xₙ, normMr )
end 

function NewArnoldi(V,A::Matrix{Float64},P::Matrix{Float64},it::Int64) #To Work on
    
    hn::Float64 = NewInnerProd(V[it],A,P,V[it])
    if length(V) == 1
        #println("V[it] :", V[it] )
        new_v::Vector{Float64} = NewMatVectProd(A,P,V[it]) - hn*V[it]
    else 
        new_v = NewMatVectProd(A,P,V[it])
        for i in range(start=1,stop=it)
            new_v = new_v - (NewInnerProd(V[i],A,P,V[it]))*V[i]
        end
    end
    h::Float64 = LinearAlgebra.norm2(new_v)
    #print("h", h, "\it")
    if !isapprox(h, 0; atol=1e-4) 
        vn = new_v/h
        #println("V[it+1] :", vn)
        V = push!(V,vn)
    else 
        V = push!(V,zeros(it))
    end 
    
    #print(size(V) ,"\t")
    return V
end

function NewHermitianLanczos(V,A::Matrix{Float64},P::Matrix{Float64},it::Int64) #To Work on
       
    if it == 1
        new_v::Vector{Float64} = NewMatVectProd(A,P,V[:,it]) 
        new_v = new_v - NewInnerProd(V[:,it],A,P,V[:,it])*V[:,it]
    else 
        new_v = NewMatVectProd(A,P,V[:,2]) - NewInnerProd(V[:,1],A,P,V[:,2])*V[:,1]
        new_v = new_v - NewInnerProd(V[:,2],A,P,V[:,2])*V[:,2]
    end
    h::Float64 = LinearAlgebra.norm2(new_v)
    #println("h", h)
    if !isapprox(h, 0; atol=1e-4) 
        vn = new_v/h
        #priln("vn :", vn)
        if it >= 2
            V[:,1] = V[:,2]
        end 
        V[:,2] = vn
    else 
        V[:,1] = V[:,2]
        V[:,2] = zeros(length(V[:,1]))
    end 
    
    #print(size(V) ,"\t")
    return V
end

function NewLanczos(V, A::Matrix{Float64}, P::Matrix{Float64}, u::Vector{Float64}, it::Int64)
    if it == 1 
        u = NewMatVectProd(A,P,V[:,it])
    end 
    α::Float64 = V[:,2]'*u #V[it]
    w::Vector{Float64} = u - α*V[:,2] #V[it]
    β::Float64 = abs(LinearAlgebra.norm2(w))
    if !isapprox(β,0;atol =1e-4)
        v::Vector{Float64} = w/β
        println("size(v) :", size(v))
        #push!(V,v)
        if it >= 2
            V[:,1] = V[:,2]
        end 
        V[:,2] = v
        u = NewMatVectProd(A, P, v) - β*V[:,1]
    end 

    return V,u
end 

function NewInnerProd(v1,A,P,v2)
    d = length(v1)
    s = Int(d/2)
    #println("(v1[1:s]')*(A')*v2[s+1:d] :", (v1[1:s]')*(A')*v2[s+1:d])
    #println("(v1[s+1:d]')*(A)*v2[1:s] :", (v1[s+1:d]')*(A)*v2[1:s])
    #println("(v1[1:s]')*P*v2[1:s] ;", (v1[1:s]')*P*v2[1:s])
    return (v1[1:s])'*(P*v2[1:s]) + (v1[1:s]')*(A')*v2[s+1:d] + (v1[s+1:d]')*(A)*v2[1:s]
end

function NewMatVectProd(A,P,v)
    #println("v :", v)
    d = length(v)
    s = Int(d/2)
    return [P*v[1:s] + (A')*v[s+1:d] ; (A)*v[1:s]]
end 

function NewUpdateT(T_diag,T_upper_diag,V,A,P,it)
    #println("NewInnerProd(V[it],A,P,V[it]) :", NewInnerProd(V[it],A,P,V[it]))
    T_diag[it] = NewInnerProd(V[:,1],A,P,V[:,1])#V[it], V[it]    
    T_upper_diag[it] = NewInnerProd(V[:,2],A,P,V[:,1])#V[it+1], V[it]

    return T_diag , T_upper_diag
end  

function IsOrthonormale(V)
    IsOrthonormal::Bool = true
    if !isapprox(V[:,2]'*V[:,1], 0 ; atol = 1e-4) 
        println("V[it+1]*V[i] :", V[:,2]'*V[:,1])
        IsOrthonormal = false
    end
    if !isapprox(V[:,2]'*V[:,2], 1; atol =1e-4 )
        println("V[it+1]*V[it+1] :", V[:,2]'*V[:,2])
        IsOrthonormal = false 
    end
    return IsOrthonormal 
end