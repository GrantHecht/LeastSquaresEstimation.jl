
abstract type AbstractNonLinearLeastSquares end

mutable struct NonLinearLeastSquares{BFVT, BFJVT}
    # Basis functions where f[i] ∀ i ∈ [1, k] should be a function 
    # which takes the state and independant
    # variable as arguments (i.e f[i](x,tₖ) where tₖ is the time at 
    # the kth epoch. For now, must be in vector form even if there
    # is only one nonlinear basis function.
    f::BFVT

    # User provided jacobian of f w.r.t. x where h[i,j] ∀ i ∈ [1, k], j ∈ [1, n]
    # computes the partial derivative of f[i] w.r.t. x[j] and takes the state
    # and independant variabels as arguments(i.e. h[i,j](x, tₖ)). For now
    # must be in matrix form even if there is only one nonlinear basis function.
    h::BFJVT

    # Jacobian matrix
    H::Matrix{Float64}

    # Estimate vector
    xhat::Vector{Float64}
    
    # Measurements
    y::Vector{Float64}

    # Measurement Weights
    W::Symmetric{Float64, Matrix{Float64}} 

    # Independant variables
    t::Vector{Float64} 

    # Tolerance
    ϵ::Float64

    # Max iterations
    iMax::Int
end

mutable struct ADNonLinearLeastSquares
    # Basis functions where f[i] ∀ i ∈ [0, n] should be a function 
    # which takes the state and independant
    # variable as arguments (i.e f[i](x,tₖ) where tₖ is the time at 
    # the kth epoch.
    f::BFVT

    # Jacobian matrix
    H::Matrix{Float64}

    # Estimate vector
    xhat::Vector{Float64}
    
    # Measurements
    y::Vector{Float64}

    # Measurement Weights
    W::Symmetric{Float64, Matrix{Float64}} 

    # Independant variables
    t::Vector{Float64}

    # Tolerance
    ϵ::Float64

    # Max iterations
    iMax::Int
end

function NonLinearLeastSquares(f::AbstractVector, h::AbstractMatrix, xhat::AbstractVector; 
                               ϵ = 1e-6, iMax = 100)
    k = length(f)
    n = length(xhat)
    kh, nh = size(h)
    if k != kh
        throw(ArgumentError("Provided jacobian does not have the correct number of rows."))
    end
    if n != nk
        throw(ArgumentError("Provided jacobian does not have the correct number of columns."))
    end

    NonLinearLeastSquares(f,h,Matrix{Float64}(undef, (0,0)), xhat, Vector{Float64}(undef, (0,0)), 
        Matrix{Float64}(undef, (0,0)), Vector{Float64}(undef, 0), ϵ, iMax)
end

function NonLinearLeastSquares(f::AbstractVector, xhat::AbstractVector; ϵ = 1e-6, iMax = 100)
    ADNonLinearLeastSquares(f, Matrix{Float64}(undef, (0,0)), xhat, Vector{Float64}(undef, (0,0)), 
        Matrix{Float64}(undef, (0,0)), Vector{Float64}(undef, 0), ϵ, iMax)
end

function FeedMeasurementBatch!(nlls::AbstractNonLinearLeastSquares, ys::AbstractVector, W::AbstractMatrix, ts::AbstractVector)
    m = length(ys)
    n = length(nlls.xhat)
    if m / length(nlls.f) != length(ts)
        throw(ArgumentError("Measurement vector and indepandant variable vector lengths are not appropriate."))
    end
    r, c = size(W)
    if r != c
        throw(ArgumentError("Weight matrix must be square!"))
    end

    # Set measurements and indepandant variable vectors
    nlls.y = ys
    nlls.W = Symmetric(W)
    nlls.t = ts

    # Size H matrix
    nlls.H = zeros(m, n)

    return nothing
end

function ComputeEstimate!(nlls::AbstractNonLinearLeastSquares)

    # Get requirements
    conv = false
    ml = length(nlls.t)
    m  = lenfth(nlls.y)
    p  = lenfth(nlls.f)
    iter = 0

    # Allocate memory 
    Jco  = Inf
    Δyc  = zeros(m)
    Δx   = zeros(n)
    fxc  = zeros(m)
    tv1  = zeros(m)
    tv2  = zeros(n)
    tm1  = zeros(m,n)
    tm2  = zeros(n,n)
    while conv == false && iter < nlls.iMax
        # Increment iteration counter 
        iter += 1

        # Fill H 
        fillH!(nlls)

        # Fill expected measurements 
        @inbounds for i in 1:ml
            for j in 1:p
                fxc[(i-1) + j] = nlls.f[j](nlls.xhat, nlls.t[i])
            end
        end

        # Compute residual and local cost function
        Δyc .= nlls.y .- fxc
        mul!(tv1, nlls.W, Δyc)
        Jc = dot(Δyc, tv1)

        # Compute change in local cost function
        δJc = abs(Jc - Jco)
        Jco = J
        if δJc < nlls.ϵ / norm(nlls.W)
            conv = true
        else
            # Compute correction
            mul!(tv2, transpose(nlls.H), tv1)
            mul!(tm1, nlls.W, nlls.H)
            mul!(tm2, transpose(nlls.H), tm1)
            fm = factorize(tm2)
            ldiv!(Δx, fm, tv2)

            # Update
            nlls.xhat .= Δx
        end
    end

    return conv
end

function fillH!(nlls::NonLinearLeastSquares)
    n  = length(nlls.xhat)
    ml = length(nlls.t)
    p  = length(nlls.f)

    # Fill 
    @inbounds for i in 1:ml
        for j in 1:p
            for k in 1:n
                nlls.H[(i - 1) + j,k] = nlls.h[j,k](nlls.xhat, nlls.t[i])
            end
        end
    end

    return nothing
end

function fillH!(nlls::ADNonLinearLeastSquares)
    n  = length(nlls.xhat)
    ml = length(nlls.t)
    p  = length(nlls.f)

    # Fill 
    @inbounds for i in 1:ml
        for j in 1:p
            grad = gradient(x -> nlls.f[j](x, nlls.t[i]), nlls.xhat)
            for k in 1:n
                H[(i - 1) + j,k] = grad[k]
            end
        end
    end   

    return nothing
end