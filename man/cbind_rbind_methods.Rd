\name{cbind_rbind_methods}
\title{cbind_rbind_methods}
\docType{methods}


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




\description{
Mimics the \code{cbind} and \code{rbind} functions.
These functions take a sequence of vectors, matrices, data-frames and/or gpu.matrix matrices and combine by columns or rows, respectively.
}


\usage{

\S4method{cbind2}{ANY,gpu.matrix.tensorflow}(x,y)
\S4method{rbind2}{ANY,gpu.matrix.torch}(x,y)
\S4method{cbind2}{gpu.matrix.tensorflow,ANY}(x,y,\dots)
\S4method{rbind2}{gpu.matrix.torch,ANY}(x,y)

}


\arguments{

\item{x}{A gpu.matrix.tensorflow o gpu.matrix.torch object}
\item{y}{A gpu.matrix.tensorflow o gpu.matrix.torch object or any other matrix class}
\item{\dots}{(generalized) vectors or matrices. These can be given as named arguments.}

}

\seealso{
\code{\link[base]{cbind}}, and \code{\link[base]{rbind}}
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

