module LeastSquaresEstimation

using LinearAlgebra
using StaticArrays
using Zygote

# Types used as constructor flags (Move to sep. file later)
abstract type SequentialEstimatorForm end
struct InformationForm <: SequentialEstimatorForm end
struct CovarianceForm <: SequentialEstimatorForm end

include("LinearLeastSquares.jl")
include("WeightedLinearLeastSquares.jl")
include("SequentialLinearLeastSquares.jl")
include("NonLinearLeastSquares.jl")

export LinearLeastSquares
export WeightedLinearLeastSquares
export SequentialLinearLeastSquares
export NonLinearLeastSquares
export InformationForm
export CovarianceForm
export FeedMeasurementBatch!
export ComputeEstimate!
export GetEstimate
export GetMeasurementEstimate

end
