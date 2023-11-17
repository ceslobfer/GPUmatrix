\name{cor_cov}
\title{Correlation, Variance and Covariance for 'GPUmatrix' objects}
\docType{methods}
\description{
Mimic of the \code{stats} functions \code{cov} and \code{cor} to compute on \code{\linkS4class{gpu.matrix}} objects: "\code{cov} and \code{cor} compute the covariance and correlation of \code{x} and \code{y} if these are vectors. If \code{x} and \code{y} are matrices then the covariances (or correlations) between the columns of \code{x} and the columns of \code{y} are computed."

\code{cov2cor} scales a covariance matrix into the corresponding correlation matrix efficiently.
}

\alias{cor}
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
\alias{cor,gpu.matrix.torch,missing,missing,missing-method}

\alias{cov2cor}
\alias{cov}
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
\S4method{cor}{ANY,gpu.matrix.tensorflow,ANY,ANY}(x,y)
\S4method{cor}{gpu.matrix.tensorflow,missing,ANY,ANY}(x,y)

\S4method{cor}{gpu.matrix.torch,ANY,ANY,ANY}(x,y)
\S4method{cor}{gpu.matrix.torch,ANY,missing,character}(x,y,method)
\S4method{cor}{gpu.matrix.torch,missing,missing,character}(x,y,method)
\S4method{cor}{ANY,gpu.matrix.torch,ANY,ANY}(x,y)
\S4method{cor}{gpu.matrix.torch,missing,ANY,ANY}(x,y,method)

\S4method{cov}{gpu.matrix.tensorflow}(x,y)
\S4method{cov}{ANY,gpu.matrix.tensorflow}(x,y)
\S4method{cov}{gpu.matrix.tensorflow,ANY}(x,y)
\S4method{cov}{gpu.matrix.tensorflow,missing}(x,y)


\S4method{cov}{gpu.matrix.torch}(x,y)
\S4method{cov}{ANY,gpu.matrix.torch}(x,y)
\S4method{cov}{gpu.matrix.torch,ANY}(x,y)
\S4method{cov}{gpu.matrix.torch,missing}(x,y)

\S4method{cov2cor}{gpu.matrix.tensorflow}(V)
\S4method{cov2cor}{gpu.matrix.torch}(V)



}


\arguments{

\item{x}{a \code{\linkS4class{gpu.matrix}}}
\item{y}{\code{NULL} (default) or a vector, matrix, data frame or \code{\linkS4class{gpu.matrix}} with compatible dimensions to x.}
\item{method}{a character string indicating which correlation coefficient (or covariance) is to be computed. One of \code{"pearson"} (default), \code{"kendall"}, or \code{"spearman"}.}
\item{V}{symmetric numeric gpu.matrix, usually positive definite such as a covariance matrix.}

}

\details{

These functions work in the same way as their counterparts in the 'stats' library.
Notice that the inputs can be either an object of class 'matrix', 'Matrix' or 'gpu.matrix'. User must be sure that the input values must be numeric (in the case of gpu.matrix objects, this is always true).

The result will be a gpu.matrix object. For more details on how to compute correlation, covariance and scaling from covariance to correlation, see \code{\link[stats]{cor}} and \code{\link[stats]{cov2cor}}.
}

\value{

The result obtained by applying these functions will be a \code{\linkS4class{gpu.matrix}} object. For each function the result will be:

- \code{cor} correlation between \code{x} and \code{y} (when two vectors are the input) or the correlation between the columns of \code{x} if \code{x} is a gpu.matrix class object.

- \code{cov} the same as \code{cor} but compute the covariance.

- \code{cov2cor} scales a covariance matrix into the corresponding correlation matrix efficiently.

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
a <- gpu.matrix(1:10)
b <- gpu.matrix(11:20)
cor(a,b)

#example taken from stats corresponding help page:
longley_gpu <- gpu.matrix(longley,type="tensorflow")
C1 <- cor(longley_gpu)
cov(longley_gpu)
cov2cor(cov(longley_gpu))

}
}
}




