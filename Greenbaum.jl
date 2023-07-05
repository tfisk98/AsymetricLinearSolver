using LinearAlgebra
#using Dates

include("PbOperations.jl")


function MINRESGreenbaum(A::Matrix{Float64},b::Vector{Float64},x₀::Vector{Float64},tol::Float64)
    #t1 = Dates.now()
    d::Int64=length(x₀)
    r₀::Vector{Float64} = b-A*x₀
    norm_r₀::Float64 = LinearAlgebra.norm2(r₀)
    V = MyHNV[]
    v1::MyHNV= MyHNV(d,r₀/norm_r₀,1)
    push!(V,v1)
    x0::MyHNV= MyHNV(d,x₀,2)
    b0::MyHNV= MyHNV(d,b,1)

    it::Int64 = 1
    norm_r::Vector{Float64} = zeros(2*d)
    norm_r[1] = norm_r₀
    α::Float64=1.0 ; β::Float64=1.0 ; η::Float64=1.0

    T= zeros(2*d) 
    S::Vector{Float64}= zeros(d+1) ; S[1]=0.
    C::Vector{Float64}= zeros(d+1) ; C[1]= 1.
    xₙ::MyHNV = x0
    #t2 = Dates.now()
    #Ρ = MyHNV[]
    ρ::MyHNV = MyHNV(d, zeros(d),2)
    ξ::Vector{Float64}= zeros(2*d); ξ[1]=1;


    #MA::Matrix{Float64} = [zeros(d,d) A ;
     #A'  zeros(d, d)]
    #Mx0::Vector{Float64} = [zeros(d) ; x0]
    #print(raw"size(Mx0.dims) =", size(Mx0),"\n")
    #Mb::Vector{Float64} = [b ; zeros(d)]
    
    #println("Initialisation :", t2-t1)
    #Orthonormale::Bool = true ; 

    #t1 = Dates.now()
    while norm_r[it] >= tol && it < 2*d
        println("it :", it)
        
        V = LanczosOptimized(A,V,it) #(d,it+1) #Mets a jour Vₙ très bizarre
        #if it ==1 
            #println("V[it]== r0/β :", (V[it]== v1 ))
        #end 

        #@assert IsOrthonormalee(V,it) "V n'est pas orthonormale"

        T[it] = inner(V[it+1],A,V[it])
        #println("T[it] :", T[it])

        #v = toVector(V[it])

        if it%2==0 
            γ = T[it-1]*C[Int(it/2)] #(it,it)
            γₙ = sqrt(γ^2 + T[it]^2)
            #println("γₙ :", γₙ)
            C[Int(it/2)+1] = γ/γₙ  ; S[Int(it/2)+1] = T[it]/γₙ ;
            norm_r[it+1] = S[Int(it/2)+1]*norm_r[it] ;
            
            ξ[it] = (γ/γₙ)*ξ[it-1]
            ξ[it+1] = (-1)*(T[it]/γₙ) * ξ[it-1] 
            #println("ξ[it] =", ξ[it])
            #println("ξ[it+1] =", ξ[it+1])

            if it == 2
                β = (-1)*C[Int(it/2)]* T[it-1]
                η = (γ/γₙ)*T[it-1] + (T[it]/γₙ)*T[it]
                #println("η :", η)
                ρ = V[it]/η

                #push!(Ρ, ρ )

                xₙ = x0 + (norm_r₀*ξ[it])*ρ
                #println("erreur réelle :", LinearAlgebra.norm2((A*xₙ-b0).vector))
                
            else 
                α = S[Int(it/2)]* T[it-1] 
                #println("C[it] :", C[Int(it/2)])
                #println("S[it] :", S[length(S)])
                #println("α :", α)
                β = (-1)*C[Int(it/2)]* T[it-1]
                

                η = (-1)*β*(γ/γₙ) + (T[it]/γₙ)*T[it]
                #println("η :", η)
                ρ = ( V[it] - α*ρ)/ η
                #println(ρ)
                #push!(Ρ, ρ )
                
                xₙ = xₙ + (norm_r₀*ξ[it])*ρ
                
                #println("xₙ :", xₙ)
                #println("A*xₙ :", (A*xₙ).vector)
                #println("erreur réelle :", LinearAlgebra.norm2((A*xₙ-b0).vector))
            end
        else 
            norm_r[it+1]=norm_r[it]

        end
        

        #println("norm_r =", norm_r[it+1])

        #ρ = v/ T[it]
        #push!(Ρ, ρ)
        it = it+1


    end 

    #t2 = Dates.now()
    #println("Fin du while")
    #println("A*xₙ :", A*xₙ.vector)
    return (it, toVector(xₙ), norm_r)
end