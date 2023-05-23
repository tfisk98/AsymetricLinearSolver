using LinearAlgebra
using IterativeSolvers 
using Dates

include("MyMinres.jl")
export MyMINRES

#Soit le probleme Ax = b 

n::Int64= 50
ϵ::Float64 = 1/n

dc = (-2)*ones(n)
sdl = ones(n-1)
Jn::Matrix{Float64} = SymTridiagonal(dc, sdl)
#Ln[1,n] = 1
#Ln[n,1] = 1
In::Matrix{Float64} = Matrix(I, n,n)
Inn::Matrix{Float64} = Matrix(I,n^2, n^2)
Bn::Matrix{Float64} = ϵ *rand(Float64, n^2, n^2)
#A = @tensor(Ln,In) + @tensor(In,Ln) + Inn + Bn
#@tensor D := Ln[i,k]*In[k,i]
#@tensor E[i,j] := In[i,k]*Ln[k,i]
F::Matrix{Float64} = LinearAlgebra.kron(Jn,In)
H::Matrix{Float64} = LinearAlgebra.kron(In,Jn)
A::Matrix{Float64} = F + H + Inn + Bn
#Test
b::Vector{Float64} = ones(n^2)
#print("length(b) =", length(b),"\n")
x0::Vector{Float64} = rand(n^2)
tol::Float64 = 10^(-4)

t1::DateTime = Dates.now()
xn::Vector{Float64} = IterativeSolvers.gmres(A,b)
t2::DateTime = Dates.now()
print("GMRES L2-error =", LinearAlgebra.norm2(b-A*xn), "\n")
println("GMRES execution time:", (t2-t1),"\n")

#Créer [0 A; At 0]

MA::Matrix{Float64} = [zeros(n^2,n^2) A ;
     A'  zeros(n^2, n^2)]
Mx0::Vector{Float64} = [zeros(n^2) ; x0]
#print(raw"size(Mx0.dims) =", size(Mx0),"\n")
Mb::Vector{Float64} = [b ; zeros(n^2)]


t1 = Dates.now()
Mxn::Vector{Float64} = IterativeSolvers.minres(MA, Mb) #Plus precis que GMRES mais tres lent
t2 = Dates.now()
print("Minres L2-error =", LinearAlgebra.norm2(Mb-MA*Mxn), "\n")
print("MINRES execution time :", (t2-t1),"\n")

t1 = Dates.now()
(itn, MMxn::Vector{Float64}, norm_r) = MyMINRES(MA, Mb, Mx0, tol)#
#print(raw"size(Mb.dims) =", size(Mb),"\n")
#print(raw"size(MMxn) =", size(MMxn),"\n")
t2 = Dates.now()
print("Nombre d'itérations :", itn,("\n"))
print("MyMinres L2-error =", LinearAlgebra.norm2(Mb-MA*MMxn), "\n")#ne marche pas 
print("MyMINRES execution time:", (t2-t1))
