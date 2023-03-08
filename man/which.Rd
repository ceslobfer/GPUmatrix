\name{which}
\title{which indices are TRUE?}

\description{

It mimics the base function \code{which}. It gives the TRUE indices of alogical object.

}


\alias{which-methods}
\alias{which,gpu.matrix.tensorflow-method}
\alias{which,gpu.matrix.torch-method}


\usage{

\S4method{which}{gpu.matrix.tensorflow}(x)
\S4method{which}{gpu.matrix.torch}(x)

}


\arguments{
\item{x}{a logical vector or array. NAs are allowed and omitted}
}


\seealso{
For more information see:
\code{\link[base]{which}}

Can be also of interest:
\code{\link[base]{which.max}} and \code{\link[base]{which.min}} that are also adapted to GPUmatrix objects.
}


\examples{
\donttest{
\dontrun{

a <- gpu.matrix(1:9,3,3,type="tensorflow")

which(a==2) #index
which(a==2,arr.ind = TRUE) #index with row and columns


}
}
}




