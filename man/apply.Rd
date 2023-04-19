\name{apply}
\title{Apply Functions over 'GPUmatrix' margins}

\description{

This function mimics the 'base' function \code{'apply'} that returns a vector or array or list of values obtained by applying a function to margins of an array or matrix.

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
\item{X}{A gpu.matrix.tensorflow or a gpu.matrix.torch object}
\item{MARGIN}{1 for rows and 2 for columns}
\item{FUN}{function to be applied in the operation}
\item{\dots}{general additional parameters. Optional arguments to FUN.}
\item{simplify}{a logical indicating whether results should be simplified if possible.}
}

\value{It returns the output corresponding to the to \code{FUN} in each iteraction to the function. For more information see the help of the base function apply.}


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












