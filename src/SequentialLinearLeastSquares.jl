
abstract type AbstractSequentialLinearLeastSquares end

mutable struct InformationFormSequentialLinearLeastSquares{BFVT} <: AbstractSequentialLinearLeastSquares
    # Basis functions linear with respect to quantities to be estimated
    # h[i] ∀ i ∈ [0, n] should be a function which takes the independant
    # variable as an argument (i.e h[i](tₖ) where tₖ is the time at 
    # the kth epoch.
    h::BFVT

    # Basis function matrix
    H::Matrix{Float64}

    # Estimate vector
    xhat::Vector{Float64}
    
    # Newest measurements
    y::Vector{Float64}

    # Newest measurement weights
    W::Symmetric{Float64, Matrix{Float64}} 

    # Newest independant variables
    t::Vector{Float64} 

    # Allocated information matricies
    Λkp1::Matrix{Float64}
    Λk::Matrix{Float64}
end

mutable struct CovarianceFormSequentialLinearLeastSquares{BFVT} <: AbstractSequentialLinearLeastSquares
    # Basis functions linear with respect to quantities to be estimated
    # h[i] ∀ i ∈ [0, n] should be a function which takes the independant
    # variable as an argument (i.e h[i](tₖ) where tₖ is the time at 
    # the kth epoch.
    h::BFVT

    # Basis function matrix
    H::Matrix{Float64}

    # Estimate vector
    xhat::Vector{Float64}
    
    # Newest measurements
    y::Vector{Float64}

    # Newest measurement weights
    W::Symmetric{Float64, Matrix{Float64}} 

    # Newest independant variables
    t::Vector{Float64} 

    # Allocated covariance matricies
    Pkp1::Matrix{Float64}
    Pk::Matrix{Float64}
end

function SequentialLinearLeastSquares(h::AbstractVector, xhat0::AbstractVector, Λ0::AbstractMatrix, form::InformationForm)
    n = length(h)
    if length(xhat0) != n
        throw(ArgumentError("Initial estimate vector length is not the same length as the basis function vector."))
    end
    r, c = size(Λ0)
    if r != n || c != n
        throw(ArgumentError("Initial information matrix is not the correct size."))
    end

    InformationFormSequentialLinearLeastSquares(h, Matrix{Float64}(undef, (0, 0)), 
        xhat0, Vector{Float64}(undef, 0), Symmetric(Matrix{Float64}(undef, (0,0))),
        Vector{Float64}(undef, 0), zeros(n,n), Λ0)
end

function SequentialLinearLeastSquares(h::AbstractVector, xhat0::AbstractVector, form::InformationForm)
    n = length(h)
    SequentialLinearLeastSquares(h, xhat0, zeros(n,n), form)
end

function SequentialLinearLeastSquares(h::AbstractVector, form::InformationForm)
    n = length(h)
    SequentialLinearLeastSquares(h, zeros(n), form)
end

function SequentialLinearLeastSquares(h::AbstractVector, xhat0::AbstractVector, P0::AbstractMatrix, form::CovarianceForm)
    n = length(h)
    if length(xhat0) != n
        throw(ArgumentError("Initial estimate vector length is not the same length as the basis function vector."))
    end
    r, c = size(P0)
    if r != n || c != n
        throw(ArgumentError("Initial covariance matrix is not the correct size."))
    end

    CovarianceFormSequentialLinearLeastSquares(h, Matrix{Float64}(undef, (0,0)),
        xhat0, Vector{Float64}(undef, 0), Symmetric(Matrix{Float64}(undef, (0,0))),
        Vector{Float64}(undef, 0), zeros(n,n), P0)
end

function FeedMeasurementBatch!(slls::AbstractSequentialLinearLeastSquares, 
                               ys::AbstractVector, W::AbstractMatrix, ts::AbstractVector)
    m = length(ys)
    n = length(slls.h)
    if m != length(ts)
        throw(ArgumentError("Measurement vector and indepandant variable vector are not the same length."))
    end
    r, c = size(W)
    if r != c
        throw(ArgumentError("Weight matrix must be square!"))
    end

    # Set measurements and indepandant variable vectors
    slls.y = ys
    slls.W = Symmetric(W)
    slls.t = ts

    # Size and fill H matrix
    H = zeros(m, n)
    @inbounds for row in 1:m
        for col in 1:n
            H[row, col] = slls.h[col](slls.t[row])
        end
    end
    slls.H = H

    return nothing
end

function ComputeEstimate!(islls::InformationFormSequentialLinearLeastSquares)
    # Compute information update
    islls.Λkp1 .= islls.Λk .+ transpose(islls.H)*islls.W*islls.H

    # Compute gain matrix
    K    = transpose(islls.H)*islls.W
    Pkp1 = factorize(islls.Λkp1)
    ldiv!(Pkp1, K)

    # Compute estimate
    islls.xhat .+= K*(islls.y .- islls.H*islls.xhat)

    # Increment information matricies
    islls.Λk .= islls.Λkp1

    return nothing
end

function ComputeEstimate!(cslls::CovarianceFormSequentialLinearLeastSquares)
    # Compute gain
    fm = factorize(cslls.H*cslls.Pk*transpose(cslls.H) .+ inv(cslls.W))
    K  = cslls.Pk*transpose(cslls.H)
    rdiv!(K, fm)

    # Update estimate
    cslls.xhat .+= K*(cslls.y .- cslls.H*cslls.xhat)

    # Update covariance
    tm = -K*cslls.H
    map(i -> tm[i,i] += 1.0, 1:length(cslls.xhat))
    cslls.Pkp1 .= tm*cslls.Pk
    cslls.Pk   .= cslls.Pkp1

    return nothing
end

function GetEstimate(slls::AbstractSequentialLinearLeastSquares)
    return slls.xhat
end

function GetMeasurementEstimate(slls::AbstractSequentialLinearLeastSquares, t::Real)
    y = 0.0
    @inbounds for i in 1:length(slls.h)
        y += slls.h[i](t)*slls.xhat[i]
    end
    return y
end
    

