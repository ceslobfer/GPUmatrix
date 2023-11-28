\name{dist}
\title{Distance Matrix Computation with GPU}
\description{
This function mimics the 'stats' function \code{dist}: 'computes and returns the distance matrix computed by using the specified distance measure to compute the distances between the rows of a data matrix.'
}

\alias{dist}
\alias{dist-methods}
\alias{dist,gpu.matrix.torch-method}

\usage{
dist(x, method = "euclidean", diag = FALSE,
    upper = FALSE, p = 2)

\S4method{dist}{gpu.matrix.torch}(x,method,diag,upper,p)
}

\arguments{
  \item{x}{a \code{\linkS4class{gpu.matrix}}.}
  \item{method}{the same as the 'stats' function \code{dist}: the distance measure to be used. Could be "euclidean", "maximum", "manhattan" or "minkowski". Note that the "canberra", "binary" methods are not included.}
  \item{diag}{the same as the 'stats' function \code{dist}: logical value indicating if the diagonal of the distances should be printed. It is set to TRUE and cannot be changed.
}
  \item{upper}{the same as the 'stats' function \code{dist}: logical value indicating whether the upper triangle of the distance matrix should be printed. It is set to TRUE and cannot be changed.}
  \item{p}{the same as the 'stats' function \code{dist}: The power of the Minkowski distance.}
}


\details{
The function mimics the 'stat' function \code{dist}. The distance measures used are (taken from \code{\link[stats]{dist}}):

euclidean: \eqn{\sqrt{\sum_i(x_i-y_i)^2}}

maximum: Maximum distance between two components of \code{x} and \code{y}.

manhattan: Absolute distance between the tow vectors.

minkowski: the \code{p} norm:  the ppth root of the sum of the ppth powers of the differences of the components.

For more details see \code{\link[stats]{dist}}.

The function \code{dist}  internally calls the corresponding function of the library torch or tensorflow (depending on the type of input gpu.matrix-class).

If the input gpu.matrix-class object is stored on the GPU, then the operations will be performed on the GPU. See \code{\link{gpu.matrix}}.


}

\seealso{
For more information see:
\code{\link[stats]{dist}}, and \code{\link[torch]{torch_cdist}}.
}

\value{
The function returns a gpu.matrix-class object with the corresponding distances between the rows of the input gpu.matrix object.
}


\examples{
\donttest{
\dontrun{

#the example compare the results with the
#'stats' function 'dist':

x <- matrix(rnorm(100), nrow = 5)

dist(x,diag = TRUE,upper = TRUE,method = "euclidean")
dist(x = as.gpu.matrix(x),method = "euclidean")

dist(x,diag = TRUE,upper = TRUE,method = "maximum")
dist(x = as.gpu.matrix(x),method = "maximum")

dist(x,diag = TRUE,upper = TRUE,method = "manhattan")
dist(x = as.gpu.matrix(x),method = "manhattan")

dist(x,diag = TRUE,upper = TRUE,method = "minkowski")
dist(x = as.gpu.matrix(x),method = "minkowski")

dist(x,diag = TRUE,upper = TRUE,method = "minkowski",p = 23)
dist(x = as.gpu.matrix(x),method = "minkowski",p = 23)

}
}
}


