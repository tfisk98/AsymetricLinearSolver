using LinearAlgebra
using TensorOperations
using IterativeSolvers 
using Dates
#using Base.Bool

include("MyMinres.jl")
export MyMINRES

#|| = TensorOperations.@tensordot

#Soit le probleme Ax = b 

n::Int64= 3
ϵ ::Float64 = 1/n

d = (-2)*ones(n)
dl = ones(n-1)

Ln = SymTridiagonal(d, dl)
#Ln[1,n] = 1
#Ln[n,1] = 1

In = Matrix(I, n,n)

Inn = Matrix(I,n^2, n^2)
Bn = ϵ *rand(Float64, n^2, n^2)

#A = @tensor(Ln,In) + @tensor(In,Ln) + Inn + Bn

#@tensor D := Ln[i,k]*In[k,i]
#@tensor E[i,j] := In[i,k]*Ln[k,i]

F = LinearAlgebra.kron(Ln,In)
H = LinearAlgebra.kron(In,Ln)

A = F + H + Inn + Bn

#Test

b = ones(n^2)
#print("length(b) =", length(b),"\n")
x0 = rand(n^2)
tol = 10^(-4)

K = 20

t1 = Dates.now()
xn = IterativeSolvers.gmres(A,b)
t2 = Dates.now()
print("GMRES L2-error =", LinearAlgebra.norm2(b-A*xn), "\n")
println("GMRES execution time:", (t2-t1),"\n")

#Créer [0 A; At 0]

MA = [zeros(n^2,n^2) A ;
     A'  zeros(n^2, n^2)]

Mx0 = [zeros(n^2) ; x0]

#print(raw"size(Mx0.dims) =", size(Mx0),"\n")

Mb = [b ; zeros(n^2)]


t1 = Dates.now()
Mxn = IterativeSolvers.minres(MA, Mb) #Plus precis que GMRES mais tres lent
t2 = Dates.now()
print("Minres L2-error =", LinearAlgebra.norm2(Mb-MA*Mxn), "\n")
print("MINRES execution time :", (t2-t1),"\n")


t1 = Dates.now()
(itn, MMxn, norm_r) = MyMINRES(MA, Mb, Mx0, tol)#
#print(raw"size(Mb.dims) =", size(Mb),"\n")
#print(raw"size(MMxn) =", size(MMxn),"\n")

t2 = Dates.now()
print("Nombre d'itérations :", itn,("\n"))
print("MyMinres L2-error =", LinearAlgebra.norm2(Mb-MA*MMxn), "\n")#ne marche pas 
print("MyMINRES execution time:", (t2-t1))
