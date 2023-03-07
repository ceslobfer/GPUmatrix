\name{matrix_general_operators_methods}
\title{matrix_general_operators_methods}
\docType{methods}

\alias{apply-methods}

\alias{density-methods}
\alias{det-methods}
\alias{chol-methods}
\alias{diag-methods}
\alias{svd-methods}
\alias{t-methods}
\alias{tail-methods}
\alias{which-methods}
\alias{show-methods}
\alias{qr-methods}
\alias{eigen-methods}
\alias{fft-methods}
\alias{head-methods}
\alias{hist-methods}


\alias{apply,gpu.matrix.tensorflow-method}
\alias{apply,gpu.matrix.torch-method}
\alias{density,gpu.matrix.tensorflow-method}
\alias{density,gpu.matrix.torch-method}
\alias{det,gpu.matrix.tensorflow-method}
\alias{det,gpu.matrix.torch-method}
\alias{chol,gpu.matrix.tensorflow-method}
\alias{chol,gpu.matrix.torch-method}
\alias{diag,gpu.matrix.tensorflow-method}
\alias{diag,gpu.matrix.torch-method}
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

\alias{eigen,gpu.matrix.tensorflow-method}
\alias{eigen,gpu.matrix.torch-method}
\alias{fft,gpu.matrix.tensorflow-method}
\alias{fft,gpu.matrix.torch-method}
\alias{head,gpu.matrix.tensorflow-method}
\alias{head,gpu.matrix.torch-method}
\alias{hist,gpu.matrix.tensorflow-method}
\alias{hist,gpu.matrix.torch-method}












\description{
Mimik of basic functions applied to matrix or Matrix class matrices. Fruther information
of these functions can be seen in its corresponding help in base.
}


\usage{

\S4method{apply}{gpu.matrix.tensorflow}(X, MARGIN, FUN, \dots, simplify)
\S4method{apply}{gpu.matrix.torch}(X, MARGIN, FUN, \dots, simplify)
\S4method{density}{gpu.matrix.tensorflow}(x)
\S4method{density}{gpu.matrix.torch}(x)
\S4method{det}{gpu.matrix.tensorflow}(x,\dots)
\S4method{det}{gpu.matrix.torch}(x,\dots)
\S4method{chol}{gpu.matrix.tensorflow}(x)
\S4method{chol}{gpu.matrix.torch}(x)


\S4method{diag}{gpu.matrix.tensorflow}(x)
\S4method{diag}{gpu.matrix.torch}(x)
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
\item{y}{A gpu.matrix.tensorflow or a gpu.matrix.torch object}
\item{object}{A gpu.matrix.tensorflow or a gpu.matrix.torch object}
\item{z}{A gpu.matrix.tensorflow or a gpu.matrix.torch object}

}
