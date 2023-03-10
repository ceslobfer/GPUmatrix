\name{matrix_ranges}
\docType{methods}
\title{Get different statistics for a matrix or get the same statistics for each row or column.}

\description{Functions to obtain the following information from a matrix or its rows (columns): maximum value,index of maximum value,minimum value,index of minimum value,mean,variance,sum of values,ranking of values.

These functions mimics the corresponding function of \code{base}, \code{matrixStats} and \code{Matrix} libraries. See also section for more details.
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
\alias{rankMatrix}
\alias{rowRanks-methods}
\alias{colRanks-methods}
\alias{rankMatrix-methods}
\alias{rowRanks,gpu.matrix.tensorflow-method}
\alias{rowRanks,gpu.matrix.torch-method}
\alias{colRanks,gpu.matrix.tensorflow-method}
\alias{colRanks,gpu.matrix.torch-method}
\alias{rankMatrix,gpu.matrix.tensorflow-method}
\alias{rankMatrix,gpu.matrix.torch-method}

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
\S4method{rankMatrix}{gpu.matrix.tensorflow}(x)
\S4method{rankMatrix}{gpu.matrix.torch}(x)

\S4method{rowSums}{gpu.matrix.tensorflow}(x)
\S4method{rowSums}{gpu.matrix.torch}(x)
\S4method{colSums}{gpu.matrix.tensorflow}(x)
\S4method{colSums}{gpu.matrix.torch}(x)
\S4method{sum}{gpu.matrix.tensorflow}(x)
\S4method{sum}{gpu.matrix.torch}(x)

}

\arguments{
\item{x}{A gpu.matrix}
}



\value{

\code{max}, \code{rowMaxs}, \code{colMaxs} calculate de maximum value of the gpu.matrix, of each row and of each column respectively. \code{which.max} determines the location of the maximum value.

\code{min}, \code{rowMinss}, \code{colMins} calculate de minimum value of the gpu.matrix, of each row and of each column respectively. \code{which.min} determines the location of the minimum value.

\code{mean}, \code{rowMeans}, \code{colMeans} calculate the mean (average) value of the gpu.matrix, of each row and of each column respectively.

\code{rowVars}, \code{colVars} calculate the variance of each row and of each column of a gpu.matrix respectively.


\code{rowRanks}, \code{colRanks}, \code{rankMatrix} get the rank of the gpu.matrix, of each row and of each column respectively.


\code{rowSums}, \code{colSums}, \code{sum} sum the value of a gpu.matrix, of each row and of each column respectively.
}


\seealso{
For more information:

\code{\link[matrixStats]{rowMaxs}},
\code{\link[matrixStats]{colMaxs}},
\code{\link[base]{max}},
\code{\link[base]{which.max}}.

\code{\link[matrixStats]{rowMins}},
\code{\link[matrixStats]{colMins}},
\code{\link[base]{min}},
\code{\link[base]{which.min}}.

\code{\link[base]{rowMeans}},
\code{\link[base]{colMeans}},
\code{\link[base]{mean}}.

\code{\link[matrixStats]{rowVars}},
\code{\link[matrixStats]{colVars}}.

\code{\link[matrixStats]{rowRanks}},
\code{\link[matrixStats]{colRanks}},
\code{\link[Matrix]{rankMatrix}}

\code{\link[base]{rowSums}},
\code{\link[base]{colSums}},
\code{\link[base]{sum}}

}





