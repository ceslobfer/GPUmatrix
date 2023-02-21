\name{aperm}
\title{aperm}
\docType{methods}



\alias{aperm-methods}

\alias{aperm,gpu.matrix.tensorflow-method}

\alias{aperm,gpu.matrix.torch-method}




\description{
Transpose a GPUmatrix by permuting its dimensions and optionally resizing it.
}


\usage{

\S4method{aperm}{gpu.matrix.tensorflow}(a,perm,\dots)
\S4method{aperm}{gpu.matrix.torch}(a,perm,\dots)

}


\arguments{

\item{a}{A gpu.matrix.tensorflow o gpu.matrix.torch object}
\item{perm}{the subscript permutation vector, usually a permutation of the integers 1:n, where n is the number of dimensions of a.}
\item{\dots}{(generalized) vectors or matrices. These can be given as named arguments.}

}
