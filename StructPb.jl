using LinearAlgebra

module MyHNVs

export MyHNV

struct MyHNV #HalfNullVect #Soit x le vecteur en question
    size:: Int64 #Il est implicitement supposé que x est en réalité de taille 2n
    vector :: Vector{Float64}
    location :: Int64 #Partie de x qui est non nulle #1 ou 2
    
    function MyHNV(size::Int64 = 1 , vector::Vector{Float64} =zeros(size), location::Int64=1)
        @assert location == 1 || location == 2 "Vecteur est seulement découpé en deux parties"
        @assert size == length(vector) "Taille de vecteur doit etre egal a size"
        new(size,vector,location)
    end
    
    """
    n=9
    A::Matrix{Float64} = rand(n,n)
    x = rand(n)
    X = MyHNV(n,x,1)
    println(A*X)
    """

end
end


