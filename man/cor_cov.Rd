\name{cor_cov}
\title{Correlation, Variance and Covariance for gpu.matrix}

\description{
\code{cov} and \code{cor} compute the covariance and correlation of x and y if these are vectors. If x and y are matrices then the covariances (or correlations) between the columns of x and the columns of y are computed.

\code{cov2cor} scales a covariance matrix into the corresponding correlation matrix efficiently.

If x and y are object of GPUmatrix, then the corresponding operations are computed by the GPU instead of the CPU.

}

\alias{cor-methods}
\alias{cor,gpu.matrix.tensorflow,ANY,ANY,ANY-method}
\alias{cor,gpu.matrix.tensorflow,ANY,missing,character-method}
\alias{cor,gpu.matrix.tensorflow,missing,missing,character-method}
\alias{cor,ANY,gpu.matrix.tensorflow,ANY,ANY-method}
\alias{cor,gpu.matrix.tensorflow,missing,ANY,ANY-method}

\alias{cor,gpu.matrix.torch,ANY,ANY,ANY-method}
\alias{cor,gpu.matrix.torch,ANY,missing,character-method}
\alias{cor,gpu.matrix.torch,missing,missing,character-method}
\alias{cor,ANY,gpu.matrix.torch,ANY,ANY-method}
\alias{cor,gpu.matrix.torch,missing,ANY,ANY-method}


\alias{cov2cor-methods}
\alias{cov-methods}
\alias{cov2cor,gpu.matrix.tensorflow-method}
\alias{cov2cor,gpu.matrix.torch-method}

\alias{cov,gpu.matrix.tensorflow-method}
\alias{cov,ANY,gpu.matrix.tensorflow-method}
\alias{cov,gpu.matrix.tensorflow,ANY-method}
\alias{cov,gpu.matrix.tensorflow,missing-method}


\alias{cov,gpu.matrix.torch-method}
\alias{cov,ANY,gpu.matrix.torch-method}
\alias{cov,gpu.matrix.torch,ANY-method}
\alias{cov,gpu.matrix.torch,missing-method}


\usage{

\S4method{cor}{gpu.matrix.tensorflow,ANY,ANY,ANY}(x,y)
\S4method{cor}{gpu.matrix.tensorflow,ANY,missing,character}(x,y,method)
\S4method{cor}{gpu.matrix.tensorflow,missing,missing,character}(x,y,method)
\S4method{cor}{ANY,gpu.matrix.tensorflow,ANY,ANY}(x,y,method)
\S4method{cor}{gpu.matrix.tensorflow,missing,ANY,ANY}(x,y,method)

\S4method{cor}{gpu.matrix.torch,ANY,ANY,ANY}(x,y)
\S4method{cor}{gpu.matrix.torch,ANY,missing,character}(x,y,method)
\S4method{cor}{gpu.matrix.torch,missing,missing,character}(x,y,method)
\S4method{cor}{ANY,gpu.matrix.torch,ANY,ANY}(x,y,method)
\S4method{cor}{gpu.matrix.torch,missing,ANY,ANY}(x,y,method)

\S4method{cov}{gpu.matrix.tensorflow}(x,y)
\S4method{cov}{ANY,gpu.matrix.tensorflow-method}(x,y)
\S4method{cov}{gpu.matrix.tensorflow,ANY-method}(x,y)
\S4method{cov}{gpu.matrix.tensorflow,missing-method}(x,y)


\S4method{cov}{gpu.matrix.torch}(x,y)
\S4method{cov}{ANY,gpu.matrix.torch-method}(x,y)
\S4method{cov}{gpu.matrix.torch,ANY-method}(x,y)
\S4method{cov}{gpu.matrix.torch,missing-method}(x,y)

\S4method{cov2cor}{gpu.matrix.tensorflow}(V)
\S4method{cov2cor}{gpu.matrix.torch}(V)



}


\arguments{

\item{x}{a numeric vector, matrix, data.frame or gpu.matrix}
\item{y}{\code{NULL} (default) or a vector, matrix, data frame or gpu.matrix with compatible dimensions to x.}
\item{method}{a character string indicating which correlation coefficient (or covariance) is to be computed. One of \code{"pearson"} (default), \code{"kendall"}, or \code{"spearman"}.}
\item{V}{symmetric numeric gpu.matrix, usually positive definite such as a covariance matrix.}

}

\value{

\code{cor} computes the correlation between x and y.

\code{cov} computes the covariance between x and y.

\code{cov2cor} scales a covariance matrix into the corresponding correlation matrix efficiently.

}

\seealso{
For more information:
\code{\link[stats]{cor}},
\code{\link[stats]{cov}},
\code{\link[stats]{cov2cor}},
}

\examples{
\donttest{
\dontrun{
a <- gpu.matrix(1:10,type="tensorflow")
b <- gpu.matrix(11:20,type="tensorflow")
cor(a,b)

#example taken from stats corresponding help page:
longley_gpu <- gpu.matrix(longley,type="tensorflow")
C1 <- cor(longley_gpu)
cov(longley_gpu)
cov2cor(cov(longley_gpu))

}
}
}




