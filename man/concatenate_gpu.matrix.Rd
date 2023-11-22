\name{concatenate_gpu.matrix}
\title{concatenate_gpu.matrix}
\docType{methods}

\description{
Mimics the 'base' function \code{'c'} to operate on \code{\linkS4class{gpu.matrix}} objects: function which "combines its arguments to form a vector. All arguments are coerced to a common type which is the type of the returned value."
In most of the cases, the returned object is of type 'numeric'.
}


\alias{c-methods}
\alias{c,gpu.matrix.tensorflow-method}
\alias{c,gpu.matrix.torch-method}
\alias{c,numMatrixLike-method}


\usage{
\S4method{c}{gpu.matrix.tensorflow}(x,\dots,recursive)
\S4method{c}{gpu.matrix.torch}(x,\dots,recursive)
\S4method{c}{numMatrixLike}(x,\dots,recursive)
}

\arguments{
\item{x}{A \code{\linkS4class{gpu.matrix}} object}
\item{\dots}{objects to be concatenated.}
\item{recursive}{The same as \code{c}: Logical. If recursive = TRUE, the function recursively descends through lists (and pairlists) combining all their elements into a vector.}

}

\value{
It will return a vector of type 'numeric' with the combined values.
}

\seealso{
See also: \code{\link[base]{c}}.
}


\examples{

\donttest{
\dontrun{

#add new value
a <- gpu.matrix(1:5,nrow=1,ncol=5)
c(a,3)

#add other vector
c(a,a)

#add value to a gpu.matrix
a <- gpu.matrix(1:9,nrow=3,ncol=3)
c(a,a)
#it will return a vector as in original c function.


}
}
}
