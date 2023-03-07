\name{concatenate_gpu.matrix}
\title{concatenate_gpu.matrix}
\docType{methods}



\alias{c-methods}
\alias{c,gpu.matrix.tensorflow-method}
\alias{c,gpu.matrix.torch-method}
\alias{c,numMatrixLike-method}




\description{
Mimic of \code{c} basic function: function which combines its arguments.
}


\usage{

\S4method{c}{gpu.matrix.tensorflow}(x,\dots,recursive)

\S4method{c}{gpu.matrix.torch}(x,\dots,recursive)

\S4method{c}{numMatrixLike}(x,\dots,recursive)

}


\arguments{

\item{x}{A gpu.matrix.tensorflow o gpu.matrix.torch object}

\item{\dots}{objects to be concatenated.}

\item{recursive}{logical. If recursive = TRUE, the function recursively descends
                          through lists (and pairlists) combining all their elements into a vector.}

}


\value{
It will return a vector with the combined values. It is equivalent to working with a matrix of class matrix.
}

\seealso{
\code{\link[base]{c}}
}


\examples{

\donttest{
\dontrun{

#add new value
a <- gpu.matrix(1:5,nrow=1,ncol=5,type = "tensorflow")
c(a,3)

#add other vector
c(a,a)

#add value to a gpu.matrix
a <- gpu.matrix(1:9,nrow=3,ncol=3,type = "tensorflow")
c(a,a)
#it will return a vector as in original c function.


}
}
}
