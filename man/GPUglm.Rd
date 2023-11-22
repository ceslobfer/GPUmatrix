\name{GPUglm}
\title{Fitting Generalized Linear Models using GPUmatrix objects}
\description{
These functions mimic the functions \code{speedglm} and \code{speedglm.wfit} of the library 'speedglm' to compute on gpu.matrix-class objects. At the same time, these functions mimic the functions \code{glm}, and \code{glm.fit} from the library 'stats' to compute on large data sets.
}
\alias{GPUglm}
\alias{glm.fit.GPU}
\usage{
glm.fit.GPU(x, y, intercept = TRUE, weights = NULL, family =
                   gaussian(), start = NULL, etastart = NULL, mustart =
                   NULL, offset = NULL, acc = 1e-08, maxit = 25, k = 2,
                   sparse = NULL, trace = FALSE, dtype = NULL, device =
                   NULL, type = NULL, ...)

GPUglm(...)
}

\arguments{
As mentioned in the description, these functions mimic \code{\link[speedglm]{speedglm}}, so almost every parameter does too. There is only three new parameters explained below.

The common parameters with \code{speedglm}:

\item{x}{the same as \code{speedglm}: the design matrix of dimension \code{n*p} where \code{n} is the number of observations and \code{p} is the number of features. \code{x} can be either a 'matrix', 'Matrix' or 'gpu.matrix-class' object.}

\item{y}{the same as \code{speedglm}: a vector of \code{n} observations. \code{y} can be either a 'matrix', 'Matrix' or 'gpu.matrix-class' object.}

\item{intercept}{the same as \code{speedglm}: Logical. If first column of \code{x} should be consider as 'intercept' (default) or not. Notice that seting this parameter TRUE or FALSE will not change the design matrix used to fit the model.}

\item{weights}{the same as \code{speedglm}: an optional vector of ‘prior weights’ to be used in the fitting process. Should be NULL (default) or a numeric vector.}

\item{family}{the same as \code{speedglm}: a description of the error distribution and link function to be used in the model. For \code{glm.fit.GPU} this can be a character string naming a family function, a family function or the result of a call to a family function. (See \code{\link[stats]{family}} for details of family functions.)}

\item{start}{the same as \code{speedglm}: starting values for the parameters in the linear prediction.}

\item{etastart}{the same as \code{speedglm}: starting values for the linear predictor.}

\item{mustart}{the same as \code{speedglm}: starting values for the vector of means.}

\item{offset}{the same as \code{speedglm}: this can be used to specify an a priori known component to be included in the linear predictor during fitting. This should be NULL or a numeric vector of length equal to the number of cases. One or more \code{\link[stats]{offset}} terms can be included in the formula instead or as well, and if more than one is specified their sum is used. See \code{\link[stats]{model.offset}}.}

\item{acc}{the same as \code{speedglm}: tolerance to be used for the estimation (by default equal to: 1e-08).}

\item{maxit}{the same as \code{speedglm}: maximum number of iterations.}

\item{k}{the same as \code{speedglm}: numeric, the penalty per parameter to be used; the default k = 2 is the classical AIC.}

\item{sparse}{if matrix \code{x} is desired to be treated as sparse. Not yet implemented.}

\item{trace}{If the user wants to see the development of the iterations. By default FALSE}

\item{...}{For \code{GPUglm}: arguments to be used to form the default control argument if it is not supplied directly.}

The \code{glm.fit.GPU} function internally initialises matrices of the 'GPUmatrix' class by calling the \code{\link[GPUmatrix]{gpu.matrix}} function. The following parameters correspond to this function:

\item{dtype}{parameter of the function \code{gpu.matrix}: "data type. User can indicate "float64", "float32" or "int" for "int64". if not specified, dtype will correspond to the input data type." }

\item{device}{parameter of the function \code{gpu.matrix}:"Only for \code{type = "torch"}. It indicates de devide to load cuda. If not indicated, 'device' will be set to 'cuda' if \code{torch::cuda_is_available()}"}

\item{type}{parameter of the function \code{gpu.matrix}: "If gpu.matrix is 'torch' (by default if type is NULL) or "tensorflow"."}

}

\details{
The \code{GPUglm} function internally calls the \code{glm} function by selecting \code{glm.fit.GPU} as the method. The input parameters of the \code{GPUglm} function are equivalent to those of the \code{glm} function.

If the gpu.matrix-class object(s) are stored on the GPU, then the operations will be performed on the GPU. See \code{\link{gpu.matrix}}.


}


\value{
Both \code{glmGPU}, and \code{glm.fit.GPU} returns an object of class "GPUglm". This object can be treated as a list. This object mimics the output of the function \code{\link[speedglm]{speedglm}}:

\item{coefficients}{the estimated coefficients.}
\item{logLik}{the log likelihood of the fitted model.}
\item{iter}{the number of iterations of IWLS used.}
\item{tol}{the maximal value of tolerance reached.}
\item{family}{the maximal value of tolerance reached.}
\item{link}{the link function used.}
\item{df}{the degrees of freedom of the model.}
\item{XTX}{the product X'X (weighted, if the case).}
\item{dispersion}{the estimated dispersion parameter of the model.}
\item{ok}{the set of column indeces of the model matrix where the model has been fitted.}
\item{rank}{the rank of the model matrix.}
\item{RSS}{the estimated residual sum of squares of the fitted model.}
\item{method}{TODO}
\item{aic}{the estimated Akaike Information Criterion.}
\item{offset}{he model offset.}
\item{sparse}{a logical value which indicates if the model matrix is sparse.}
\item{deviance}{the estimated deviance of the fitted model.}
\item{nulldf}{the degrees of freedom of the null model.}
\item{nulldev}{the estimated deviance of the null model.}
\item{ngoodobs}{the number of non-zero weighted observations.}
\item{n}{the number of observations.}
\item{intercept}{a logical value which indicates if an intercept has been used.}
\item{convergence}{a logical value which indicates if convergence was reached.}
\item{terms}{the terms object used.}
\item{call}{the matched call.}
\item{xlevels}{(where relevant) a record of the levels of the factors used in fitting.}




}
\seealso{
See also: \code{\link[speedglm]{speedglm}} and \code{\link[stats]{glm}}.
}
\examples{

\donttest{
\dontrun{
# These examples show that the way to call
# and work with these functions is the same
# as with the corresponding functions
# of the `speedglm' library.

# linear model:
x <- matrix(rnorm(100*50),nrow = 100)
y <- x %*% rnorm(50) + rnorm(100)
s1 <- glm.fit.GPU(x=Matrix(x),y=as.gpu.matrix(y))

#poisson and binomial:

m <- 1000
n <- 100
x <- matrix(runif(m*n),m,n)
sol <- rnorm(n)
y <- rbinom(m, 1, prob = plogis(x%*%sol))
s1 <- glm.fit.GPU(x,y,family = binomial(),dtype="float32")
s2 <- glm.fit.GPU(x,y,family = poisson(),dtype="float64", device = "cpu")

}
}

}

