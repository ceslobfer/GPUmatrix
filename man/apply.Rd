\name{apply}
\title{Apply Functions over 'gpu.matrix-class' margins}

\description{

This function mimics the 'base' function \code{'apply'} to operate on gpu.matrix-class objects: "It returns a vector, an array or a list of values obtained by applying a function to margins of a GPUmatrix."

}
\alias{apply}
\alias{apply-methods}
\alias{apply,gpu.matrix.tensorflow-method}
\alias{apply,gpu.matrix.torch-method}

\usage{
\S4method{apply}{gpu.matrix.tensorflow}(X, MARGIN, FUN, \dots, simplify)
\S4method{apply}{gpu.matrix.torch}(X, MARGIN, FUN, \dots, simplify)
}

\arguments{
\item{X}{A \code{\linkS4class{gpu.matrix}} object}
\item{MARGIN}{1 for rows and 2 for columns}
\item{FUN}{function to be applied in the operation}
\item{\dots}{general additional parameters. Optional arguments to FUN.}
\item{simplify}{a logical indicating whether results should be simplified if possible.}
}

\details{
\code{FUN} is found by a call to \code{\link[base]{match.fun}} as done in the base function \code{\link[base]{apply}}. Internally, apply will use the functions implemented to work with objects from the GPUmatrix library.

As in \code{\link[base]{apply}}, the arguments in \code{...} cannot have the same name as any of the other arguments to ensure possible errors.

}

\value{The results of mimics the base function \code{\link[base]{apply}}:

If each call to \code{FUN} returns a vector of length \code{n}, and \code{simplify} is \code{TRUE}, then apply returns a matrix of \code{n} rows and \code{dim(X)[MARGIN]} columns if \code{n > 1}. If \code{n} equals 1, \code{apply} returns a vector.

If the calls to \code{FUN} return vectors of different lengths, or if \code{simplify} is \code{FALSE}, apply returns a list of length \code{dim(X)[MARGIN]}.

In all cases the result is coerced by as.vector to one of the basic vector types before the dimensions are set, so that (for example) factor results will be coerced to a character array.

For more details see \code{\link[base]{apply}}
}


\seealso{
For more information see:
\code{\link[base]{apply}}
}

\examples{

if(installTorch){

  a <- gpu.matrix(rnorm(9),3,3)

  apply(a, 1, mean) #computes the mean of each row
  apply(a, 2, mean) #computes the mean of each column

}




}












