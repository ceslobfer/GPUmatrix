\name{cbind_rbind_methods}
\title{cbind_rbind_methods}
\docType{methods}

\description{
Mimics the 'base' functions \code{'cbind'} and \code{'rbind'} to operate on \code{\linkS4class{gpu.matrix}} objects. The 'base' functions \code{'cbind'} and \code{'rbind'} internally call the methods \code{cbind2} and \code{rbind2}.

Therefore, ss done in \code{\link[Matrix]{cbind2}} of the package 'Matrix', we have defined in 'GPUmatrix' the methods \code{cbind2} and \code{rbind2} to operate on \code{\linkS4class{gpu.matrix}} objects too.
}


\alias{cbind2}
\alias{rbind2}
\alias{cbind2-methods}
\alias{rbind2-methods}
\alias{cbind2,ANY,gpu.matrix.tensorflow-method}
\alias{cbind2,ANY,gpu.matrix.torch-method}
\alias{cbind2,gpu.matrix.tensorflow,ANY-method}
\alias{cbind2,gpu.matrix.torch,ANY-method}
\alias{rbind2,ANY,gpu.matrix.tensorflow-method}
\alias{rbind2,ANY,gpu.matrix.torch-method}
\alias{rbind2,gpu.matrix.tensorflow,ANY-method}
\alias{rbind2,gpu.matrix.torch,ANY-method}



\usage{

\S4method{cbind2}{ANY,gpu.matrix.tensorflow}(x,y)
\S4method{rbind2}{ANY,gpu.matrix.torch}(x,y)
\S4method{cbind2}{gpu.matrix.tensorflow,ANY}(x,y,\dots)
\S4method{rbind2}{gpu.matrix.torch,ANY}(x,y)

}


\arguments{

\item{x,y}{a \code{\linkS4class{gpu.matrix}} object or any other matrix class.}
\item{\dots}{(generalized) vectors or matrices. These can be given as named arguments.}

}

\value{
The result of using these functions is equivalent to using the basic \code{cbind} and \code{rbind} functions. For more details see \code{\link[base]{cbind}}.

Note that if one of the input values is a gpu.matrix-class object, then the output will also be a gpu.matrix-class object.

The data type of the values of the resulting gpu.matrix-class object (corresponding to the \code{dtype} parameter of the gpu.matrix function) is the one that allows the integration of all input values. That is, if you call \code{cbind(a,b)} where \code{a} is a gpu.matrix-class object with values of "int32" and \code{b} is a gpu.matrix-class with values of "float64", the result will be a gpu.matrix-class with values of "float64".
}



\seealso{
\code{\link[base]{cbind}}, \code{\link[base]{rbind}}, \code{\link[Matrix]{cbind2}}
}

\examples{
\donttest{
\dontrun{

a <- gpu.matrix(1:9,nrow=3,ncol=3)

#add new row
newrow <- c(1,2,3)
a <- rbind2(a,newrow)

#add new column
newcolumn <- c(1,2,3,4)
a <- cbind(a,newcolumn)

#add new rows from other gpu.marix
b <- gpu.matrix(1:16,nrow=4,ncol=4)
d <- rbind(a,b)

#add new columns from other gpu.marix
b <- gpu.matrix(1:16,nrow=4,ncol=4)
d <- cbind(a,b)


}
}
}

