using LinearAlgebra




function MyMINRESUpdated(A::Matrix{Float64},b::Vector{Float64},x₀::Vector{Float64},tol::Float64)
    d::Int16=length(x₀)
    @assert A' - A == zeros(d, d) "Non symétrique"
    r₀::Vector{Float64} = b-A*x₀
    norm_r₀::Float64 = LinearAlgebra.norm2(r₀) 
    V::Matrix{Float64} = zeros(d,1)
    Vₙ::Matrix{Float64} = V
    V[:,1] = r₀/norm_r₀
    it::Int16 = 1
    norm_r::Float64 = norm_r₀

    
    T::Matrix{Float64} = zeros(it,it)
    T[it,it] = V[:,1]'*A*V[:,1]
    #println("T :",T)
    Lₙ::Matrix{Float64}, Qₙ::Matrix{Float64} = LinearAlgebra.lq(T')
    Gₙ::Matrix{Float64} = zeros(it,it)#Matrice de Givens 
    xₙ::Vector{Float64} = x₀
    Hₙ::Matrix{Float64} = zeros(it,it)
    Cₙ::Matrix{Float64} = zeros(2,2)
    e₁::Vector{Float64} = zeros(it) ; e₁[1]=1 ;


    while norm_r >= tol 
        Vₙ = V #(it,it)
        V = HermitianLanczosUpdated2(A,V,it) #(d,it+1)
        T = V'*A*Vₙ #(it+1, it)
        Tₙ::Matrix{Float64} = T[1:it,:]
        if !isapprox(Tridiagonal(Tₙ), Tₙ; atol=1e-4)
            println("Tₙ :", Tₙ)
            break
        end
        Hₙ = Tₙ*Qₙ' #(it,it)
        γₙ::Float64 = sqrt(Hₙ[it,it]^2 + T[it+1,it]^2^2)
        cₙ::Float64 = Hₙ[it,it]/γₙ  ; sₙ::Float64= T[it+1,it]^2/γₙ ;
        Cₙ = [cₙ sₙ ; sₙ -cₙ] #(2,2)
        if it == 1 
            Gₙ = Cₙ
        else 
            Gₙ = [I(it-1) zeros(it-1,2) ;
                    zeros(2,it-1)   Cₙ] #(it+1,it+1)
        end 
        Qₙ = Gₙ*[Qₙ zeros(it) ; zeros(it)' 1 ]  #(it+1, it+1)
        #if it<= 5 
            #println("Tₙ:", Tₙ)
            #println("Hₙ:", Hₙ)
            #println("Cₙ:", Cₙ)
            #println("Gₙ:", Gₙ)
            #println("Qₙ:", Qₙ)
        #end 
        @assert isapprox((Qₙ')*Qₙ, I;atol=1e-4) "Qₙ doit être orthogonale"

        Sₙ::Matrix{Float64} = T'*Qₙ' #(it,it+1)
        #println("size(Sₙ):", size(Sₙ))
        Lₙ = Sₙ*[I(it) ; zeros(it)'] #(it, it)

        e₁ = zeros(it+1) ; e₁[1]=1
        eₘ::Vector{Float64} = zeros(it+1) ; eₘ[it+1]=1 
        norm_r = abs(norm_r₀*(eₘ'*(Qₙ)*e₁))
        println("norm_r =", norm_r)

        it = it+1

        #xₙ= Vₙ*(inv(Rₙ))*(norm_r₀*([Matrix(I,it-1,it-1) zeros(it-1,1)]*Qₙ*e₁)) + x₀
        #nrm  = LinearAlgebra.norm2(b - A*xₙ)
        #println("|b-Axn| :", nrm)

    end 
    
    #println("T :", T)
    e₁ = zeros(it); e₁[1] =1;
    xₙ= Vₙ*(inv(Lₙ'))*(norm_r₀*([Matrix(I,it-1,it-1) zeros(it-1,1)]*Qₙ*e₁)) +x₀
    return (it-1, xₙ, norm_r)
end 




function HermitianLanczosUpdated2(A::Matrix{Float64},V::Matrix{Float64},it::Int16) #To Work on
       
    hn::Float64 = (V[:,it]')*A*V[:,it] 
    if it == 1
        new_v::Vector{Float64} = A*V[:,it] - hn*V[:,it]
    else 
        new_v = A*V[:,it] - hn*V[:,it] - ((V[:,it-1]')*A*V[:,it])*V[:,it-1]
    end
    h::Float64 = LinearAlgebra.norm2(new_v)
    #print("h", h, "\it")
    if !isapprox(h, 0; atol=1e-4) 
        vn = new_v/h
        V = [V  vn ]
    else 
        V = [V zeros(it)]
    end 
    
    print(size(V) ,"\t")
    return V
end