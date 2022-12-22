using LinearAlgebra: svd, svdvals, Diagonal, norm
using MIRTjim: jim
using Plots; default(markerstrokecolor=:auto)

m = [1 NaN 3; 4 5 6;7 8 9;10 11 12]
Y=[isnan(i) ? 0 : i for i in m]
omega= Y .!= 0
nucnorm = (X) -> sum(svdvals(X))
costfun = (X,beta) -> 0.5 * norm(X[omega]-Y[omega])^2 + beta * nucnorm(X)
SVST = (X,beta) -> begin
    U,s,V = svd(X)
    sthresh = max.(s .- beta,0)
    return U * Diagonal(sthresh) * V'
end;
# Apply ISTA (Iterative Soft-Thresholding Algorithm)
niter = 50
beta = 0.01 # chosen by trial-and-error here
function lrmc_ista(Y)
	X = copy(Y)
	Xold = copy(X)
	cost_ista = zeros(niter+1)
	cost_ista[1] = costfun(X,beta)
	for k=1:niter
    	X[omega] = Y[omega]
    	X = SVST(X,beta)
    	cost_ista[k+1] = costfun(X,beta)
	end
	return X, cost_ista
end

function lrmc_fista(Y)
	X = copy(Y)
	Z = copy(X)
	Xold = copy(X)
	told = 1
	cost_fista = zeros(niter+1)
	cost_fista[1] = costfun(X,beta)
	for k=1:niter
    	Z[omega] = Y[omega]
    	X = SVST(Z,beta)
    	t = (1 + sqrt(1+4*told^2))/2
    	Z = X + ((told-1)/t)*(X-Xold)
    	Xold = X
    	told = t
    	cost_fista[k+1] = costfun(X,beta) # comment out to speed-up
	end
	return X, cost_fista
end

function lrmc_admm(Y)
	X = copy(Y)
	Z = zeros(size(X))
	L = zeros(size(X))
	mu = beta # choice of this parameter can greatly affect convergence rate
	cost_admm = zeros(niter+1)
	cost_admm[1] = costfun(X,beta)
	for k=1:niter
    	Z = SVST(X + L, beta / mu)
    	X = (Y + mu * (Z - L)) ./ (mu .+ omega)
        
    	L = L + X - Z
    	cost_admm[k+1] = costfun(X,beta) # comment out to speed-up
	end
	return X, cost_admm
end
X, cost_ista = lrmc_admm(Y)
println(X)