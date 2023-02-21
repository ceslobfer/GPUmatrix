\name{matrix_general_operators_methods}
\title{matrix_general_operators_methods}
\docType{methods}



\alias{rowMaxs-methods}
\alias{rowMeans-methods}
\alias{rowMins-methods}
\alias{rowRanks-methods}
\alias{rowSums-methods}
\alias{rowVars-methods}
\alias{rownames-methods}
\alias{apply-methods}
\alias{cov2cor-methods}
\alias{cov-methods}
\alias{density-methods}
\alias{det-methods}
\alias{chol-methods}
\alias{colMaxs-methods}
\alias{colMeans-methods}
\alias{colMins-methods}
\alias{colRanks-methods}
\alias{colSums-methods}
\alias{colVars-methods}
\alias{colnames-methods}
\alias{diag-methods}
\alias{dim-methods}
\alias{dimnames-methods}
\alias{sum-methods}
\alias{svd-methods}
\alias{t-methods}
\alias{tail-methods}
\alias{which-methods}
\alias{show-methods}
\alias{qr-methods}
\alias{rankMatrix-methods}
\alias{length-methods}
\alias{max-methods}
\alias{mean-methods}
\alias{min-methods}
\alias{ncol-methods}
\alias{nrow-methods}
\alias{eigen-methods}
\alias{fft-methods}
\alias{head-methods}
\alias{hist-methods}
\alias{which.max-methods}
\alias{which.min-methods}

\alias{rowMaxs,gpu.matrix.tensorflow-method}
\alias{rowMaxs,gpu.matrix.torch-method}
\alias{rowMeans,gpu.matrix.tensorflow-method}
\alias{rowMeans,gpu.matrix.torch-method}
\alias{rowMins,gpu.matrix.tensorflow-method}
\alias{rowMins,gpu.matrix.torch-method}
\alias{rowRanks,gpu.matrix.tensorflow-method}
\alias{rowRanks,gpu.matrix.torch-method}
\alias{rowSums,gpu.matrix.tensorflow-method}
\alias{rowSums,gpu.matrix.torch-method}
\alias{rowVars,gpu.matrix.tensorflow-method}
\alias{rowVars,gpu.matrix.torch-method}
\alias{rownames,gpu.matrix.tensorflow-method}
\alias{rownames,gpu.matrix.torch-method}
\alias{apply,gpu.matrix.tensorflow-method}
\alias{apply,gpu.matrix.torch-method}
\alias{cov2cor,gpu.matrix.tensorflow-method}
\alias{cov2cor,gpu.matrix.torch-method}
\alias{cov,gpu.matrix.tensorflow-method}
\alias{cov,gpu.matrix.torch-method}
\alias{density,gpu.matrix.tensorflow-method}
\alias{density,gpu.matrix.torch-method}
\alias{det,gpu.matrix.tensorflow-method}
\alias{det,gpu.matrix.torch-method}
\alias{chol,gpu.matrix.tensorflow-method}
\alias{chol,gpu.matrix.torch-method}
\alias{colMaxs,gpu.matrix.tensorflow-method}
\alias{colMaxs,gpu.matrix.torch-method}
\alias{colMeans,gpu.matrix.tensorflow-method}
\alias{colMeans,gpu.matrix.torch-method}
\alias{colMins,gpu.matrix.tensorflow-method}
\alias{colMins,gpu.matrix.torch-method}
\alias{colRanks,gpu.matrix.tensorflow-method}
\alias{colRanks,gpu.matrix.torch-method}
\alias{colSums,gpu.matrix.tensorflow-method}
\alias{colSums,gpu.matrix.torch-method}
\alias{colVars,gpu.matrix.tensorflow-method}
\alias{colVars,gpu.matrix.torch-method}
\alias{colnames,gpu.matrix.tensorflow-method}
\alias{colnames,gpu.matrix.torch-method}
\alias{diag,gpu.matrix.tensorflow-method}
\alias{diag,gpu.matrix.torch-method}
\alias{dim,gpu.matrix.tensorflow-method}
\alias{dim,gpu.matrix.torch-method}
\alias{dimnames,gpu.matrix.tensorflow-method}
\alias{dimnames,gpu.matrix.torch-method}
\alias{sum,gpu.matrix.tensorflow-method}
\alias{sum,gpu.matrix.torch-method}
\alias{svd,gpu.matrix.tensorflow-method}
\alias{svd,gpu.matrix.torch-method}
\alias{t,gpu.matrix.tensorflow-method}
\alias{t,gpu.matrix.torch-method}
\alias{tail,gpu.matrix.tensorflow-method}
\alias{tail,gpu.matrix.torch-method}
\alias{which,gpu.matrix.tensorflow-method}
\alias{which,gpu.matrix.torch-method}
\alias{show,gpu.matrix.tensorflow-method}
\alias{show,gpu.matrix.torch-method}
\alias{qr,gpu.matrix.tensorflow-method}
\alias{qr,gpu.matrix.torch-method}
\alias{rankMatrix,gpu.matrix.tensorflow-method}
\alias{rankMatrix,gpu.matrix.torch-method}
\alias{length,gpu.matrix.tensorflow-method}
\alias{length,gpu.matrix.torch-method}
\alias{max,gpu.matrix.tensorflow-method}
\alias{max,gpu.matrix.torch-method}
\alias{mean,gpu.matrix.tensorflow-method}
\alias{mean,gpu.matrix.torch-method}
\alias{min,gpu.matrix.tensorflow-method}
\alias{min,gpu.matrix.torch-method}
\alias{ncol,gpu.matrix.tensorflow-method}
\alias{ncol,gpu.matrix.torch-method}
\alias{nrow,gpu.matrix.tensorflow-method}
\alias{nrow,gpu.matrix.torch-method}
\alias{eigen,gpu.matrix.tensorflow-method}
\alias{eigen,gpu.matrix.torch-method}
\alias{fft,gpu.matrix.tensorflow-method}
\alias{fft,gpu.matrix.torch-method}
\alias{head,gpu.matrix.tensorflow-method}
\alias{head,gpu.matrix.torch-method}
\alias{hist,gpu.matrix.tensorflow-method}
\alias{hist,gpu.matrix.torch-method}
\alias{which.max,gpu.matrix.tensorflow-method}
\alias{which.max,gpu.matrix.torch-method}
\alias{which.min,gpu.matrix.tensorflow-method}
\alias{which.min,gpu.matrix.torch-method}








\description{
Mimik of basic functions applied to matrix or Matrix class matrices. Fruther information
of these functions can be seen in its corresponding help in base.
}


\usage{
\S4method{rowMaxs}{gpu.matrix.tensorflow}(x)
\S4method{rowMaxs}{gpu.matrix.torch}(x)
\S4method{rowMeans}{gpu.matrix.tensorflow}(x)
\S4method{rowMeans}{gpu.matrix.torch}(x)
\S4method{rowMins}{gpu.matrix.tensorflow}(x)
\S4method{rowMins}{gpu.matrix.torch}(x)
\S4method{rowRanks}{gpu.matrix.tensorflow}(x)
\S4method{rowRanks}{gpu.matrix.torch}(x)
\S4method{rowSums}{gpu.matrix.tensorflow}(x)
\S4method{rowSums}{gpu.matrix.torch}(x)
\S4method{rowVars}{gpu.matrix.tensorflow}(x)
\S4method{rowVars}{gpu.matrix.torch}(x)
\S4method{rownames}{gpu.matrix.tensorflow}(x)
\S4method{rownames}{gpu.matrix.torch}(x)
\S4method{apply}{gpu.matrix.tensorflow}(X, MARGIN, FUN, \dots, simplify)
\S4method{apply}{gpu.matrix.torch}(X, MARGIN, FUN, \dots, simplify)
\S4method{cov2cor}{gpu.matrix.tensorflow}(V)
\S4method{cov2cor}{gpu.matrix.torch}(V)
\S4method{cov}{gpu.matrix.tensorflow}(x,y)
\S4method{cov}{gpu.matrix.torch}(x,y)
\S4method{density}{gpu.matrix.tensorflow}(x)
\S4method{density}{gpu.matrix.torch}(x)
\S4method{det}{gpu.matrix.tensorflow}(x,\dots)
\S4method{det}{gpu.matrix.torch}(x,\dots)
\S4method{chol}{gpu.matrix.tensorflow}(x)
\S4method{chol}{gpu.matrix.torch}(x)
\S4method{colMaxs}{gpu.matrix.tensorflow}(x)
\S4method{colMaxs}{gpu.matrix.torch}(x)
\S4method{colMeans}{gpu.matrix.tensorflow}(x)
\S4method{colMeans}{gpu.matrix.torch}(x)
\S4method{colMins}{gpu.matrix.tensorflow}(x)
\S4method{colMins}{gpu.matrix.torch}(x)
\S4method{colRanks}{gpu.matrix.tensorflow}(x)
\S4method{colRanks}{gpu.matrix.torch}(x)
\S4method{colSums}{gpu.matrix.tensorflow}(x)
\S4method{colSums}{gpu.matrix.torch}(x)
\S4method{colVars}{gpu.matrix.tensorflow}(x)
\S4method{colVars}{gpu.matrix.torch}(x)
\S4method{colnames}{gpu.matrix.tensorflow}(x)
\S4method{colnames}{gpu.matrix.torch}(x)
\S4method{diag}{gpu.matrix.tensorflow}(x)
\S4method{diag}{gpu.matrix.torch}(x)
\S4method{dim}{gpu.matrix.tensorflow}(x)
\S4method{dim}{gpu.matrix.torch}(x)
\S4method{dimnames}{gpu.matrix.tensorflow}(x)
\S4method{dimnames}{gpu.matrix.torch}(x)
\S4method{sum}{gpu.matrix.tensorflow}(x)
\S4method{sum}{gpu.matrix.torch}(x)
\S4method{svd}{gpu.matrix.tensorflow}(x)
\S4method{svd}{gpu.matrix.torch}(x)
\S4method{t}{gpu.matrix.tensorflow}(x)
\S4method{t}{gpu.matrix.torch}(x)
\S4method{tail}{gpu.matrix.tensorflow}(x,\dots)
\S4method{tail}{gpu.matrix.torch}(x,\dots)
\S4method{which}{gpu.matrix.tensorflow}(x)
\S4method{which}{gpu.matrix.torch}(x)
\S4method{show}{gpu.matrix.tensorflow}(object)
\S4method{show}{gpu.matrix.torch}(object)
\S4method{qr}{gpu.matrix.tensorflow}(x)
\S4method{qr}{gpu.matrix.torch}(x)
\S4method{rankMatrix}{gpu.matrix.tensorflow}(x)
\S4method{rankMatrix}{gpu.matrix.torch}(x)
\S4method{length}{gpu.matrix.tensorflow}(x)
\S4method{length}{gpu.matrix.torch}(x)
\S4method{max}{gpu.matrix.tensorflow}(x)
\S4method{max}{gpu.matrix.torch}(x)
\S4method{mean}{gpu.matrix.tensorflow}(x)
\S4method{mean}{gpu.matrix.torch}(x)
\S4method{min}{gpu.matrix.tensorflow}(x)
\S4method{min}{gpu.matrix.torch}(x)
\S4method{ncol}{gpu.matrix.tensorflow}(x)
\S4method{ncol}{gpu.matrix.torch}(x)
\S4method{nrow}{gpu.matrix.tensorflow}(x)
\S4method{nrow}{gpu.matrix.torch}(x)
\S4method{eigen}{gpu.matrix.tensorflow}(x)
\S4method{eigen}{gpu.matrix.torch}(x)
\S4method{fft}{gpu.matrix.tensorflow}(z)
\S4method{fft}{gpu.matrix.torch}(z)
\S4method{head}{gpu.matrix.tensorflow}(x,\dots)
\S4method{head}{gpu.matrix.torch}(x,\dots)
\S4method{hist}{gpu.matrix.tensorflow}(x,\dots)
\S4method{hist}{gpu.matrix.torch}(x,\dots)

}


\arguments{

\item{x}{A gpu.matrix.tensorflow or gpu.matrix.torch object}
\item{X}{A gpu.matrix.tensorflow or a gpu.matrix.torch object}
\item{MARGIN}{1 for rows and 2 for columns}
\item{FUN}{function to be applied in the operation}
\item{\dots}{general additional parameters. Optional arguments to FUN.}
\item{simplify}{a logical indicating whether results should be simplified if possible.}
\item{V}{A gpu.matrix.tensorflow or a gpu.matrix.torch object}
\item{y}{A gpu.matrix.tensorflow or a gpu.matrix.torch object}
\item{object}{A gpu.matrix.tensorflow or a gpu.matrix.torch object}
\item{z}{A gpu.matrix.tensorflow or a gpu.matrix.torch object}

}
