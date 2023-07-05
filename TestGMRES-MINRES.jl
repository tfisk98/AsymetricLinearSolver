using LinearAlgebra
using IterativeSolvers 
using Dates
using Plots


#include("MyGmresUpdated.jl")
include("MINRESOptimized.jl")
include("MyMinresUpdated.jl")
include("MyMinresUpdated2.jl")
include("Greenbaum.jl")
include("ArnoldiVsLanczos.jl")
include("NouveauProblemeGreenbaum.jl")

#Soit le probleme Ax = b 

n::Int64= 30
ϵ::Float64 = 1/n #Pb operations

dc::Vector{Float64} = 4*ones(n) #Pb Operations
sdl::Vector{Float64} = -ones(n-1) #Pb Opeations
Jn::Matrix{Float64} = SymTridiagonal(dc, sdl)
#Ln[1,n] = 1
#Ln[n,1] = 1
In::Matrix{Float64} = Matrix(I, n,n)
Inn::Matrix{Float64} = Matrix(I,n^2, n^2)
Bn::Matrix{Float64} = ϵ *rand(Float64, n^2, n^2) #Pb Operation *
#A = @tensor(Ln,In) + @tensor(In,Ln) + Inn + Bn
#@tensor D := Ln[i,k]*In[k,i]
#@tensor E[i,j] := In[i,k]*Ln[k,i]
F::Matrix{Float64} = LinearAlgebra.kron(Jn,In)
H::Matrix{Float64} = LinearAlgebra.kron(In,Jn)
A::Matrix{Float64} = F + H + Inn + Bn

d::Vector{Float64} = 4*ones(n^2)
sd::Vector{Float64} = -ones(n^2-1)
L::Matrix{Float64} = SymTridiagonal(d,sd) 
#P::Matrix{Float64} = inv(L)
P = L


#P::Matrix{Float64} = zeros(n^2,n^2)
#P::Matrix{Float64} = A'*A
#P= Matrix(Diagonal(diag(A)))
#Test
b::Vector{Float64} = ones(n^2)
#print("length(b) =", length(b),"\n")
x0::Vector{Float64} = zeros(n^2)
tol::Float64 = 1e-4
#K::Int16 = 200
"""
t1 = Dates.now()
xn::Vector{Float64} = IterativeSolvers.gmres(A,b)
t2 = Dates.now()
print("GMRES L2-error =", LinearAlgebra.norm2(b-A*xn), "\n")
println("GMRES execution time:", (t2-t1),"\n")
"""

#Créer [0 A; At 0]

MA::Matrix{Float64} = [zeros(n^2,n^2) A ;
     A'  zeros(n^2, n^2)]
MAt::Matrix{Float64} = [P A' ;
     A  zeros(n^2, n^2)]
Mx0::Vector{Float64} = [zeros(n^2) ; x0]
#print(raw"size(Mx0.dims) =", size(Mx0),"\n")
Mb::Vector{Float64} = [b ; zeros(n^2)]
Mb2::Vector{Float64} = [zeros(n^2) ; b ]

"""
t1 = Dates.now()
Mxn::Vector{Float64} = IterativeSolvers.minres(MA, Mb) #Plus precis que GMRES mais tres lent
t2 = Dates.now()
print("Minres L2-error =", LinearAlgebra.norm2(Mb-MA*Mxn), "\n")
print("MINRES execution time :", (t2-t1),"\n")


itn = ArnoldiVsLanczos(A, P, x0, x0, b)
println("Consistance Arnoldi-Lanczos :", itn)



#Optmz = MyMINRESUpdated(A, b, x0, tol)

t1 = Dates.now()
#(itn, MMxn::Vector{Float64}, norm_r) = MyMINRESUpdated(A, b, x0, tol)
Updt2 = MyMINRESUpdated2(A, b, x0, tol)
t2 = Dates.now()
println("Nombre d'itérations :", Updt2[1])
println("MyMINRESUpdated L2-error =", LinearAlgebra.norm2(Mb-MA*Updt2[2]))#ne marche pas 
println("MyMINRESUpdated execution time:", (t2-t1))
"""
t1 = Dates.now()
#(itn, MMxn::Vector{Float64}, norm_r) = MyMINRESUpdated(A, b, x0, tol)
Gb = MINRESGreenbaum(A, b, x0, tol)
t2 = Dates.now()
println("Nombre d'itérations :", Gb[1])
println("MyMINRESUpdated L2-error =", LinearAlgebra.norm2(Mb-MA*Gb[2]))#ne marche pas 
println("MyMINRESUpdated execution time:", (t2-t1))



t1 = Dates.now()
#(itn, MMxn::Vector{Float64}, norm_r) = MyMINRESUpdated(A, b, x0, tol)
NpbGb = NouveauProblemeGreenbaum(A, P, Mx0, Mb2, tol)
t2 = Dates.now()
println("Nombre d'itérations :", NpbGb[1])
println("MyMINRESUpdated L2-error =", LinearAlgebra.norm2(Mb2-(MAt)*NpbGb[2]))#ne marche pas 
println("MyMINRESUpdated execution time:", (t2-t1))
#println("Solution :", NpbGb[2])

x = range(start=1,stop=NpbGb[1], length=NpbGb[1])
y = NpbGb[3][1:NpbGb[1]]
println("x :", x)
println("y :", y)
plot(x, y)

z = range(0, 10, length=100)
t = sin.(z)
plot(z, t)
"""
#t1 = Dates.now()
#Optmz = MyMINRESUpdated(A, b, x0, tol)
#println("Mxn :", Optmz[2])
#t2 = Dates.now()
#println("Nombre d'itérations :", Optmz[1])
#println("MyMINRESUpdated L2-error =", LinearAlgebra.norm2(Mb-MA*Optmz[2])) 
#println("MyMINRESUpdated execution time:", (t2-t1))


if !(Updt2 == Optmz)
     #("Equal solutions :",  (Updt2[2] == Optmz[2]))
     #println("diff sol :", LinearAlgebra.norm2(Updt2[2]- Optmz[2]))
     #("Equal residuals :", (Updt2[3] == Optmz[3]))
     println("Equal Qₙ's :", isapprox(Updt2[4]- Optmz[4], zeros(Updt2[1] +1, Updt2[1] +1 ); atol=1e-4))#Problème de signe sur Qₙ
     println("Diff Qₙ matrices :", Updt2[4]- Optmz[4])
     println("Diff Rₙ matrices :", Updt2[5]- Optmz[5])
     println("Diff Rₙ :",maximum(abs2, Updt2[5]- Optmz[5]))
     println("Diff Qₙ :",maximum(abs2, Updt2[4]- Optmz[4]))
     #("Updt2 Qₙ :",Updt2[4])
     #println("Updt2 Rₙ :", Updt2[5])
     #println("Optmz Rₙ :", Optmz[5])
end 

t1 = Dates.now()
#(itn, MMxn::Vector{Float64}, norm_r) = MyMINRESUpdated(A, b, x0, tol)
Gupdt = MyGMRESUpdated(MA, Mb, Mx0, tol)
t2 = Dates.now()
println("Nombre d'itérations :", Gupdt[1])
println("MyMINRESUpdated L2-error =", LinearAlgebra.norm2(Mb-MA*Gupdt[2]))#ne marche pas 
println("MyMINRESUpdated execution time:", (t2-t1))
"""

