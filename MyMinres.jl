using LinearAlgebra

#Erreur d'arrondis sur les calculs de prod scalaire pour l'assemblage de Tn

#|| = LinearAlgebra.norm2
 
# n=n+1 a la fin de while 
#lq sur la transposée de Tn et non tn
#v de HermitianLanczos bien initialisé



function MyMINRES(A::Matrix{Float64},b::Vector{Float64},x0::Vector{Float64},tol::Float64)
    d = size(A)[1]
    @assert A' - A == zeros(d, d) "Non symétrique"
    r::Vector{Float64} = b - A*x0
    it::Int16 = 1 
    norm_r::Float64 = LinearAlgebra.norm2(r)
    V::Matrix{Float64} = zeros(d,1) 
    V[:,1]::Vector{Float64} = (r/norm_r)
    Vₙ::Matrix{Float64} = V
    norm_rO::Float64 = norm_r
    Ln, Qn = lq(ones(1,1)) 
    while(norm_r >= tol) #&& n <= 10)
        #AlgoOrthonomalisationLanczos
        Vₙ = V #Vₙ (d,n)
        #print(size(V) ,"\t")
        V = HermitianLanczos(A,r,it) #(d, n+1 )
        #print("V =", V,"\n")
        @assert V[:,1:it] == Vₙ "Problème Hermitian Lanczos"
        #print(size(V) ,"\t")
        Tn = (V')*A*Vₙ #(n+1,n) #Problème ICI
        Testn = Tn[1:it,:]
        #print("Tn :", Tn ,"\n") #Erreur d'arrondis sur les calculs de prod scalaire
        #@assert isapprox(Tridiagonal(Testn), Testn; atol=1e-4)  "Testn doit être tridiagonal"
        if !isapprox(Tridiagonal(Testn), Testn; atol=1e-2)
            println("Tn :", Tn)
            break
        end
        #print("size Tn =", size(Tn), "\t")
        (l,q) = LinearAlgebra.lq(Tn')#lq sur la transposée de Tn et non tn
        Ln = l #(n,n) 
        #println(size(Ln))
        #print("Ln =", Ln ,"\n")
        #print("size Ln =", size(Ln), "\t")
        Qn = q#(n+1,n+1)
        @assert isapprox((Qn')*Qn, I;atol=1e-4) "Qn doit être orthogonale"
        #print("size Qn =", size(Qn), "\t")
        e_un = zeros(it+1) ; e_un[1] = 1 
        e_nPlusUn = zeros(it+1) ;e_nPlusUn[it+1]= 1 
        norm_r = abs(norm_rO*e_nPlusUn'*(Qn)*e_un)
        print("norm_r =", norm_r,"\n")
        it = it+1

    end

    print("Ln :", size(Ln) ,"\t")
    print("Vn :", size(Vₙ),"\t")
    println("Qn :",size(Qn) )
    e::Vector{Float64} = zeros(it); e[1] =1;
    println("e :",size(e) )
    println("it", it)
    #Ln_t = inverse(transpose(Ln))
    x::Vector{Float64}= Vₙ*(inv(Ln'))*(norm_rO*([Matrix(I,it-1,it-1) zeros(it-1,1)]*Qn*e))


    return (it,x,norm_r)

end

function HermitianLanczos(A::Matrix{Float64},r::Vector{Float64},it::Int16) #To Work on
    d = length(r)
    v::Matrix{Float64} = zeros(d,it+1)
    norm_r::Float64  = LinearAlgebra.norm2(r)
    v[:,1]::Vector{Float64} = (r/norm_r)

    for j in range(1,it)
       
        hjj::Float64 = (v[:,j]')*A*v[:,j] 
        if j == 1
            new_v::Vector{Float64} = A*v[:,j] - hjj*v[:,j]
        else 
            new_v = A*v[:,j] - hjj*v[:,j] - ((v[:,j-1]')*A*v[:,j])*v[:,j-1]
        end
        h::Float64 = LinearAlgebra.norm2(new_v)
        #print("h", h, "\n")
        if !isapprox(h, 0; atol=1e-4) 
            v[:,j+1] = new_v/h
        end 
    end 
    print(size(v) ,"\t")
    return v 
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