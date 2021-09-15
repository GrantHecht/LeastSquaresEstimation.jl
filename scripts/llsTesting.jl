using LeastSquaresEstimation
using Plots

function main()
    # Define basis function vector
    h = [t->sin(t), t->cos(t), t->t, t->cos(t)*sin(t), t->t^2]

    # Instantiate Linear Least Squares Estimator
    llse = LinearLeastSquares(h)

    # Generate Measurements
    m    = 101
    y(t) = 0.3*sin(t) + 0.5*cos(t) + 0.1*t + sqrt(0.001)*randn()

    ts = zeros(m)
    ys = zeros(m)
    @inbounds for i in 1:m
        if i != 1
            ts[i] = ts[i-1] + 0.1
        end
        ys[i] = y(ts[i])
    end

    # Feed estimator batch of measurements
    FeedMeasurementBatch!(llse, ys, ts)

    # Compute estimate 
    ComputeEstimate!(llse)

    # Compute estimated measurements and plot 
    yhat = zeros(m)
    @inbounds for i in 1:m
        yhat[i] = GetMeasurementEstimate(llse, ts[i])
    end

    # Display 
    println("Estimated parameters:")
    display(GetEstimate(llse))

    # Plot generated and estimated measurements 
    plot(ts, ys; label = "Simulated measurements")
    plot!(ts, yhat; label = "Estimated measurements")
end

main()