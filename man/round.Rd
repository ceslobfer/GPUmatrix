\name{round}
\title{rounding of numers}
\docType{methods}


\description{
It mimics the base function \code{'round'} that rounds the values in its first argument to the specified number of decimal places (default 0).
}


\alias{round}
\alias{round-methods}
\alias{round,gpu.matrix.tensorflow,ANY-method}
\alias{round,gpu.matrix.torch,missing-method}
\alias{round,gpu.matrix.torch,numeric-method}
\alias{round,gpu.matrix.tensorflow,missing-method}
\alias{round,gpu.matrix.tensorflow,numeric-method}


\usage{
\S4method{round}{gpu.matrix.tensorflow,ANY}(x)
\S4method{round}{gpu.matrix.torch,missing}(x,digits)
\S4method{round}{gpu.matrix.torch,numeric}(x,digits)
\S4method{round}{gpu.matrix.tensorflow,missing}(x,digits)
\S4method{round}{gpu.matrix.tensorflow,numeric}(x,digits)
}


\arguments{

\item{x}{a numeric GPUmatrix object.}
\item{digits}{integer indicating the number of decimal places (round) or significant digits (signif) to be used.}

}


\seealso{
\code{\link[base]{round}}
}


\examples{
\donttest{
\dontrun{

a <- gpu.matrix(rnorm(9),3,3)
round(a,digits = 3) #round to the third digit


}
}
}



