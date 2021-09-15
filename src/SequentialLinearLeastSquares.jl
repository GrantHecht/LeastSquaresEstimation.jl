
mutable struct InformationFormSequentialLinearLeastSquares{BFVT}

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

function SequentialLinearLeastSquares(h::AbstractVector, xhat0::AbstractVector, form::SequentialEstimatorForm)
    n = length(h)
    InformationFormSequentialLinearLeastSquares(h, Matrix{Float64}(undef, (0, 0)), 
        Vector{Float64}(undef, n), Vector{Float64}(undef, 0), 
        Symmetric(Matrix{Float64}(undef, (0,0))),Vector{Float64}(undef, 0),
        zeros(n,n), zeros(n,n))
end

function FeedMeasurementBatch!(islls::InformationFormSequentialLinearLeastSquares, 
                               ys::AbstractVector, W::AbstractMatrix, ts::AbstractVector)
    m = length(ys)
    n = length(islls.h)
    if m != length(ts)
        throw(ArgumentError("Measurement vector and indepandant variable vector are not the same length."))
    end
    r, c = size(W)
    if r != c
        throw(ArgumentError("Weight matrix must be square!"))
    end

    # Set measurements and indepandant variable vectors
    islls.y = ys
    islls.W = Symmetric(W)
    islls.t = ts

    # Size and fill H matrix
    H = zeros(m, n)
    @inbounds for row in 1:m
        for col in 1:n
            H[row, col] = islls.h[col](islls.t[row])
        end
    end
    islls.H = H

    return nothing
end

function ComputeEstimate!(islls::InformationFormSequentialLinearLeastSquares)
    # Compute information update
    islls.Λkp1 .= islls.Λk .+ transpose(islls.H)*islls.W*islls.H

    # Compute gain matrix
    K    = zeros(length(islls.h), length(islls.y))
    Pkp1 = factorize(islls.Λkp1)
    ldiv!(K, Pkp1, transpose(islls.H)*islls.W)

    # Compute estimate
    islls.xhat .+= K*(islls.y - islls.H*islls.xhat)

    # Increment information matricies
    islls.Λk .= islls.Λkp1

    return nothing
end

function GetEstimate(islls::InformationFormSequentialLinearLeastSquares)
    return islls.xhat
end

function GetMeasurementEstimate(islls::InformationFormSequentialLinearLeastSquares, t::Real)
    y = 0.0
    @inbounds for i in 1:length(islls.h)
        y += islls.h[i](t)*islls.xhat[i]
    end
    return y
end
    

