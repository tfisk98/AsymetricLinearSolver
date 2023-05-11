using LinearAlgebra
using Core

#Erreur d'arrondis sur les calculs de prod scalaire pour l'assemblage de Tn

#|| = LinearAlgebra.norm2
 
# n=n+1 a la fin de while 
#lq sur la transposée de Tn et non tn
#v de HermitianLanczos bien initialisé



function MyMINRES(A,b,x0,tol)
    d = size(A)[1]
    @assert A' - A == zeros(d, d) "Non symétrique"
    r = b - A*x0
    n = 1 
    norm_r = LinearAlgebra.norm2(r)
    V = zeros(d,1) 
    V[:,1] = (r/norm_r)
    norm_rO = norm_r
    while(norm_r >= tol && n <= 10)
        #AlgoOrthonomalisationLanczos
        Vₙ = V #Vₙ (d,n)
        #print(size(V) ,"\t")
        V = HermitianLanczos(A,r,n) #(d, n+1 )
        #print("V =", V,"\n")
        @assert V[:,1:n] == Vₙ "Problème Hermitian Lanczos"
        #print(size(V) ,"\t")
        Tn = (V')*A*Vₙ #(n+1,n) #Problème ICI
        Testn = Tn[1:n,:]
        print("Tn :", Tn ,"\t") #Erreur d'arrondis sur les calculs de prod scalaire
        @assert Tridiagonal(Testn) - Testn == zeros(n,n)  "Testn doit être tridiagonal"
        #print("size Tn =", size(Tn), "\t")
        (Fl,Fq) = LinearAlgebra.lq(Tn')#lq sur la transposée de Tn et non tn
        Ln = Fl #(n,n) 
        #print("Ln :", Ln ,"\t")
        #print("size Ln =", size(Ln), "\t")
        Qn = Fq#(n+1,n+1)
        #print("size Qn =", size(Qn), "\t")
        e_un = zeros(n+1,1) ; e_un[1,1] = 1 
        e_nPlusUn = zeros(1,n+1) ;e_nPlusUn[1,n+1]= 1 

        norm_r = abs(norm_rO*(e_nPlusUn*(Qn)*e_un)[1,1])
        #print("norm_r =", norm_r,"\n")
        n = n+1

    end

    print("Ln :", Ln ,"\t")
    e = zeros(n,1); e[1,1] =1;
    Ln_t = inverse(transpose(Ln))
    x= W*Ln_t(norm_rO*([Matrix(I,n,n); zeros(n,1)]*Qn*e))
    return (n,x,norm_r)

end

function HermitianLanczos(A,r,n) #To Work on
    d = length(r)
    v = zeros(d,n+1)
    norm_r  = LinearAlgebra.norm2(r)
    v[:,1] = (r/norm_r)

    for j in range(1,n)
       
        hjj = (v[:,j]')*A*v[:,j] 
        if j == 1
            new_v = A*v[:,j] - hjj*v[:,j]
        else 
            new_v = A*v[:,j] - hjj*v[:,j] - ((v[:,j-1]')*A*v[:,j])*v[:,j-1]
        end
        h = LinearAlgebra.norm2(new_v)
        if h != 0 
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

