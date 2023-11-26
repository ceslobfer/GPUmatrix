\name{round}
\title{rounding of numers}
\docType{methods}


\description{
It mimics the base function \code{'round'} to operate on gpu.matrix-class objects. This function rounds the values in its first argument to the specified number of decimal places (default 0).
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

\item{x}{a \code{\linkS4class{gpu.matrix}}.}
\item{digits}{integer indicating the number of decimal places (round) or significant digits (signif) to be used.}

}

\details{
The function \code{round} internally calls the corresponding function of the library torch or tensorflow (depending on the type of input gpu.matrix-class).

The behaveour of the function mimics the 'base' function \code{round}. Note that for rounding off a 5, the function will consider "go to the even digit". Therefore, \code{round(2.5) = 2} and \code{round(3.5) = 4}. For more details see \code{\link[base]{round}}, and \code{\link[torch]{torch_round}}.

If the input gpu.matrix-class object is stored on the GPU, then the operations will be performed on the GPU. See \code{\link{gpu.matrix}}.
}

\value{
The function will return a gpu.matrix-class object with the rounded values.
}

\seealso{
\code{\link[base]{round}}, and \code{\link[torch]{torch_round}}.
}


\examples{
\donttest{
\dontrun{

a <- gpu.matrix(rnorm(9),3,3)
round(a,digits = 3) #round to the third digit


}
}
}



