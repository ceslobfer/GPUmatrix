\name{[-methods}
\title{Methods for "[": Extraction or Subsetting in Package 'GPUmatrix'}
\docType{methods}
%
\alias{[-methods}
%
\alias{[,gpu.matrix.tensorflow,index,index,ANY-methods}
\alias{[,gpu.matrix.tensorflow,index,missing,ANY-methods}
\alias{[,gpu.matrix.tensorflow,matrix,missing,ANY-methods}
\alias{[,gpu.matrix.tensorflow,missing,index,ANY-methods}
\alias{[,gpu.matrix.torch,index,index,ANY-methods}
\alias{[,gpu.matrix.torch,index,missing,ANY-methods}
\alias{[,gpu.matrix.torch,matrix,missing,ANY-methods}
\alias{[,gpu.matrix.torch,missing,index,ANY-methods}

%
\description{
  Methods for \code{"["}, i.e., extraction or subsetting mostly of
  matrices, in package \pkg{GPUmatrix}.
}

\arguments{
\item{x}{a gpu.matrix}
\item{i}{index for rows}
\item{j}{index for columns}
\item{value}{vecotr}
}

\section{Methods}{
  There are more than these:
  \describe{
    \item{x = "gpu.matrix.torch", i = "index", j = "index",value=ANY}{ ... }
    \item{x = "gpu.matrix.tensorflow", i = "index", j = "index",value=ANY}{ ... }
    \item{x = "gpu.matrix.torch", i = "matrix", j = "missing",value=ANY}{ ... }
    \item{x = "gpu.matrix.tensorflow", i = "matrix", j = "missing",value=ANY}{ ... }
    \item{x = "gpu.matrix.torch", i = "missing", j = "inidex",value=ANY}{ ... }
    \item{x = "gpu.matrix.tensorflow", i = "missing", j = "index",value=ANY}{ ... }

  }
}
\seealso{
  \code{\link{[<--methods}} for sub\emph{assign}ment to \code{"GPUmatrix"}
  objects.
  \code{\link{Extract}} about the standard extraction.
}
\examples{

a <- gpu.matrix(as.numeric(1:9),nrow=3,ncol=3)
a[1,1]
}
\keyword{methods}
\keyword{array}
