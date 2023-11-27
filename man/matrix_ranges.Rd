\name{matrix_ranges}
\docType{methods}
\title{Get different statistics for a gpu.matrix-class.}

\description{
Functions to summarise different values of a gpu.matrix-class object by rows or columns. Specifically: the maximum value, the index of the maximum value, the minimum value, the index of the minimum value, the mean, the variance, the sum of the values and the rank of the values.

These functions mimic the corresponding function of \code{'base'}, \code{'matrixStats'} and \code{'Matrix'} libraries.
}

\alias{rowMaxs}
\alias{rowMaxs-methods}
\alias{rowMaxs,gpu.matrix.tensorflow-method}
\alias{rowMaxs,gpu.matrix.torch-method}

\alias{colMaxs}
\alias{colMaxs-methods}
\alias{colMaxs,gpu.matrix.tensorflow-method}
\alias{colMaxs,gpu.matrix.torch-method}
\alias{max-methods}
\alias{which.max-methods}
\alias{max,gpu.matrix.tensorflow-method}
\alias{max,gpu.matrix.torch-method}
\alias{which.max,gpu.matrix.tensorflow-method}
\alias{which.max,gpu.matrix.torch-method}


\alias{rowMins}
\alias{colMins}
\alias{min}
\alias{which.min}

\alias{rowMins-methods}
\alias{colMins-methods}
\alias{min-methods}
\alias{which.min-methods}
\alias{rowMins,gpu.matrix.tensorflow-method}
\alias{rowMins,gpu.matrix.torch-method}
\alias{colMins,gpu.matrix.tensorflow-method}
\alias{colMins,gpu.matrix.torch-method}
\alias{min,gpu.matrix.tensorflow-method}
\alias{min,gpu.matrix.torch-method}
\alias{which.min,gpu.matrix.tensorflow-method}
\alias{which.min,gpu.matrix.torch-method}

\alias{rowMeans}
\alias{colMeans}
\alias{mean}
\alias{rowMeans-methods}
\alias{colMeans-methods}
\alias{mean-methods}
\alias{rowMeans,gpu.matrix.tensorflow-method}
\alias{rowMeans,gpu.matrix.torch-method}
\alias{colMeans,gpu.matrix.tensorflow-method}
\alias{colMeans,gpu.matrix.torch-method}
\alias{mean,gpu.matrix.tensorflow-method}
\alias{mean,gpu.matrix.torch-method}

\alias{rowVars}
\alias{colVars}
\alias{rowVars-methods}
\alias{colVars-methods}
\alias{rowVars,gpu.matrix.tensorflow-method}
\alias{rowVars,gpu.matrix.torch-method}
\alias{colVars,gpu.matrix.tensorflow-method}
\alias{colVars,gpu.matrix.torch-method}

\alias{rowRanks}
\alias{colRanks}
\alias{rowRanks-methods}
\alias{colRanks-methods}
\alias{rowRanks,gpu.matrix.tensorflow-method}
\alias{rowRanks,gpu.matrix.torch-method}
\alias{colRanks,gpu.matrix.tensorflow-method}
\alias{colRanks,gpu.matrix.torch-method}

\alias{rowSums}
\alias{colSums}
\alias{sum}
\alias{rowSums-methods}
\alias{colSums-methods}
\alias{sum-methods}
\alias{rowSums,gpu.matrix.tensorflow-method}
\alias{rowSums,gpu.matrix.torch-method}
\alias{colSums,gpu.matrix.tensorflow-method}
\alias{colSums,gpu.matrix.torch-method}
\alias{sum,gpu.matrix.tensorflow-method}
\alias{sum,gpu.matrix.torch-method}



\usage{
\S4method{rowMaxs}{gpu.matrix.tensorflow}(x)
\S4method{rowMaxs}{gpu.matrix.torch}(x)
\S4method{colMaxs}{gpu.matrix.tensorflow}(x)
\S4method{colMaxs}{gpu.matrix.torch}(x)
\S4method{max}{gpu.matrix.tensorflow}(x)
\S4method{max}{gpu.matrix.torch}(x)

\S4method{rowMins}{gpu.matrix.tensorflow}(x)
\S4method{rowMins}{gpu.matrix.torch}(x)
\S4method{colMins}{gpu.matrix.tensorflow}(x)
\S4method{colMins}{gpu.matrix.torch}(x)
\S4method{min}{gpu.matrix.tensorflow}(x)
\S4method{min}{gpu.matrix.torch}(x)

\S4method{rowMeans}{gpu.matrix.tensorflow}(x)
\S4method{rowMeans}{gpu.matrix.torch}(x)
\S4method{colMeans}{gpu.matrix.tensorflow}(x)
\S4method{colMeans}{gpu.matrix.torch}(x)
\S4method{mean}{gpu.matrix.tensorflow}(x)
\S4method{mean}{gpu.matrix.torch}(x)

\S4method{rowVars}{gpu.matrix.tensorflow}(x)
\S4method{rowVars}{gpu.matrix.torch}(x)
\S4method{colVars}{gpu.matrix.tensorflow}(x)
\S4method{colVars}{gpu.matrix.torch}(x)

\S4method{rowRanks}{gpu.matrix.tensorflow}(x)
\S4method{rowRanks}{gpu.matrix.torch}(x)
\S4method{colRanks}{gpu.matrix.tensorflow}(x)
\S4method{colRanks}{gpu.matrix.torch}(x)


\S4method{rowSums}{gpu.matrix.tensorflow}(x)
\S4method{rowSums}{gpu.matrix.torch}(x)
\S4method{colSums}{gpu.matrix.tensorflow}(x)
\S4method{colSums}{gpu.matrix.torch}(x)
\S4method{sum}{gpu.matrix.tensorflow}(x)
\S4method{sum}{gpu.matrix.torch}(x)

}

\arguments{
\item{x}{a \code{\linkS4class{gpu.matrix}}.}
}

\value{

\code{max}, \code{rowMaxs}, \code{colMaxs} calculate the maximum value of a gpu.matrix-class object, of each row and of each column respectively. \code{which.max} determines the location of the maximum value.

\code{min}, \code{rowMins}, \code{colMins} calculate the minimum value of a gpu.matrix-class object, of each row and of each column respectively. \code{which.min} determines the location of the minimum value.

\code{mean}, \code{rowMeans}, \code{colMeans} calculate the mean (average) value of a gpu.matrix-class object, of each row and of each column respectively.

\code{rowVars}, \code{colVars} calculate the variance of each row and of each column of a gpu.matrix-class object respectively.


\code{rowRanks}, \code{colRanks}: given a gpu.matrix-class object, these functions return a gpu.matrix which rearranges each row and each column into ascending respectively.


\code{rowSums}, \code{colSums}, \code{sum} sum the value of a a gpu.matrix-class object, of each row and of each column respectively.
}


\details{

The value returned by almost each function is a numeric vector stored in the CPU. Only the function \code{rowRanks}, \code{colRanks}, and \code{sum} return a gpu.matrix-class object.

These functions internally calls the corresponding function of the library torch or tensorflow (depending on the type of input gpu.matrix-class).
If the input gpu.matrix-class object is stored on the GPU, then the operations will be performed on the GPU. See \code{\link{gpu.matrix}}.


}




\seealso{
For more information:

\code{\link[matrixStats]{rowMaxs}},
\code{\link[matrixStats]{colMaxs}},
\code{\link[base]{max}},
\code{\link[base]{which.max}}, and \code{\link[torch]{torch_max}}.

\code{\link[matrixStats]{rowMins}},
\code{\link[matrixStats]{colMins}},
\code{\link[base]{min}},
\code{\link[base]{which.min}}, and \code{\link[torch]{torch_min}}.

\code{\link[base]{rowMeans}},
\code{\link[base]{colMeans}},
\code{\link[base]{mean}}, and \code{\link[torch]{torch_mean}}.

\code{\link[matrixStats]{rowVars}},
\code{\link[matrixStats]{colVars}}, and \code{\link[torch]{torch_var}}.

\code{\link[matrixStats]{rowRanks}},
\code{\link[matrixStats]{colRanks}}, and \code{\link[torch]{torch_argsort}}.

\code{\link[base]{rowSums}},
\code{\link[base]{colSums}},
\code{\link[base]{sum}}, and \code{\link[torch]{torch_sum}}.

}


\examples{
\donttest{
\dontrun{
a <- gpu.matrix(rnorm(9),3,3)

#the maximum value of a:
max(a)

#maximum of value in each row of a:
rowMaxs(a)

#maximum value in each column of a:
colMaxs(a)

#index of the maximum value of a:
which.max(a)

#minimum value of a:
min(a)

#minimum value in each row of a:
rowMins(a)

#minimum value in each column of a:
colMins(a)

#index of the minimum value in a:
which.min(a)

#mean of a:
mean(a)

#mean of each row of a:
rowMeans(a)

#mean of each column of a:
colMeans(a)

#variance of each row of a:
rowVars(a)

#variance of each column of a:
colVars(a)

#sum of all values of a:
sum(a)

#sum of each fow of a:
rowSums(a)

#sum of each column of a:
colSums(a)

#ranking of each row of a:
rowRanks(a)

#ranking of each columna of a:
colRanks(a)
}
}
}



