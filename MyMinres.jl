using LinearAlgebra

#Erreur d'arrondis sur les calculs de prod scalaire pour l'assemblage de Tn

#|| = LinearAlgebra.norm2
 
# n=n+1 a la fin de while 
#lq sur la transposée de Tn et non tn
#v de HermitianLanczos bien initialisé



function MyMINRES(A::Matrix{Float64},b::Vector{Float64},x0::Vector{Float64},tol::Float64)
    d::Int16 = size(A)[1]
    @assert A' - A == zeros(d, d) "Non symétrique"
    r::Vector{Float64} = b - A*x0
    it::Int16 = 1 #Juqu'a 65k iterations
    norm_r::Float64 = LinearAlgebra.norm2(r)
    V::Matrix{Float64} = zeros(d,1) 
    V[:,1]::Vector{Float64} = (r/norm_r)
    Vₙ::Matrix{Float64} = V
    norm_rO::Float64 = norm_r
    Ln::Matrix{Float64} = zeros(1,1); #Qn::Matrix{Float64}= zeros(3,3)
    while(norm_r >= tol) #&& n <= 10)
        #AlgoOrthonomalisationLanczos
        Vₙ = V #Vₙ (d,n)
        #print(size(V) ,"\t")
        V = HermitianLanczos(A,V,it) #(d, n+1 )
        #print("V =", V,"\n")
        @assert V[:,1:it] == Vₙ "Problème Hermitian Lanczos"
        #print(size(V) ,"\t")
        T::Matrix{Float64} = (V')*A*Vₙ #(n+1,n) #Problème ICI
        Tn::Matrix{Float64} = T[1:it,:]
        #print("Tn :", Tn ,"\n") #Erreur d'arrondis sur les calculs de prod scalaire
        #@assert isapprox(Tridiagonal(Testn), Testn; atol=1e-4)  "Testn doit être tridiagonal"
        if !isapprox(Tridiagonal(Tn), Tn; atol=1e-2)
            println("Tn :", Tn)
            break
        end
        #print("size Tn =", size(Tn), "\t")
        (l,q) = LinearAlgebra.lq(T')#lq sur la transposée de Tn et non tn
        Ln = l #(n,n) 
        #println(size(Ln))
        #print("Ln =", Ln ,"\n")
        #print("size Ln =", size(Ln), "\t")
        Qn = q#(n+1,n+1)
        if it<= 5 
            #println("Tn:", Tn)
            #println("Ln:", Ln)
            println("Qn:", Qn)
        end 
        @assert isapprox((Qn')*Qn, I;atol=1e-4) "Qn doit être orthogonale"
        #print("size Qn =", size(Qn), "\t")
        e_un::Vector{Float64} = zeros(it+1) ; e_un[1] = 1 
        e_nPlusUn::Vector{Float64} = zeros(it+1) ;e_nPlusUn[it+1]= 1 
        norm_r = abs(norm_rO*e_nPlusUn'*(Qn)*e_un)
        print("norm_r =", norm_r,"\n")
        it = it+1

        xₙ::Vector{Float64}= Vₙ*(inv(Ln'))*(norm_rO*([Matrix(I,it-1,it-1) zeros(it-1,1)]*Qn*e_un)) + x0
        nrm  = LinearAlgebra.norm2(b - A*xₙ)
        println("|b-Axn| :", nrm)

    end

    print("Ln :", size(Ln) ,"\t")
    print("Vn :", size(Vₙ),"\t")
    println("Qn :",size(Qn) )
    e::Vector{Float64} = zeros(it); e[1] =1;
    println("e :",size(e) )
    println("it", it)
    #Ln_t = inverse(transpose(Ln))
    x= Vₙ*(inv(Ln'))*(norm_rO*([Matrix(I,it-1,it-1) zeros(it-1,1)]*Qn*e)) + x0


    return (it,x,norm_r)

end

function HermitianLanczos(A::Matrix{Float64},V::Matrix{Float64},it::Int16) #To Work on
       
    hit::Float64 = (V[:,it]')*A*V[:,it] 
    if it == 1
        new_v::Vector{Float64} = A*V[:,it] - hit*V[:,it]
    else 
        new_v = A*V[:,it] - hit*V[:,it] - ((V[:,it-1]')*A*V[:,it])*V[:,it-1]
    end
    h::Float64 = LinearAlgebra.norm2(new_v)
    #print("h", h, "\n")
    if !isapprox(h, 0; atol=1e-4) 
        vit = new_v/h
    else 
        return [V zeros(length(V[:,1]))]
    end
    
    V = [V  vit ]
    print(size(V) ,"\t")
    return V
end
        


"""
F = LinearALgebra.LU(transpose(Tn))
Ln = F.L
Un = U.L
Lt = transpose(L)
L_t = inverse(Lt)




hen = A*W - V*Tn
        hn = hen[1,n+1]
        c = Tn[n,n]/ sqrt(hn^2 + Tn[n,n]^2 )
        s = hn /sqrt(hn^2 + Tn[n,n]^2 )
        Omegan = [I(n-1,n-1) zeros(n-1,2); zeros(2,n-1) [c, s; -s, c]]
        Qn = [Q zeros(n,1); zeros(1,n) 1]
        Q = Omegan*Qn
        e_un = zeros(n+1) ; e_un[1] = 1 
        e_nPlusUn = zeros(1,n+1) ;e_nPlusUn[1,n+1]= 1 

        norm_r = norm_r0*(e_nPlusUn*Q*e_un)
"""

"""
pour n>=2
new_v = A*V[:,n] - (V[:,n]')*A*V[:,n] - ((v[:,n-1]')*A*v[:,n])*v[:,n-1]

h::{Float64} = LinearAlgebra.norm2(new_v)
if !isapprox(h, 0; atol=1e-4)  
    v = new_v/ h
    V= [V v]
end
"""