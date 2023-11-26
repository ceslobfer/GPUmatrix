\name{dim_and_names}
\title{Number of rows and columns and its corresponding names}

\description{
These functions mimic the 'base' functions \code{rownames}, \code{colnames}, \code{dimnames}, \code{dim}, \code{length}, \code{ncol}, \code{nrow} to operate on gpu.matrix-class objects.

The "\code{dim} family functions" set or get the dimension of a gpu.matrix-class object.

The "\code{rownames} and \code{colnames} family functions" set or get the corresponding names of rows and columns of a gpu.matrix-class object.
}

\alias{rownames}
\alias{colnames}
\alias{dimnames}
\alias{dim}
\alias{length}
\alias{ncol}
\alias{nrow}


\alias{rownames-methods}
\alias{colnames-methods}
\alias{dimnames-methods}
\alias{dim-methods}
\alias{length-methods}
\alias{ncol-methods}
\alias{nrow-methods}
\alias{rownames,gpu.matrix.tensorflow-method}
\alias{rownames,gpu.matrix.torch-method}
\alias{colnames,gpu.matrix.tensorflow-method}
\alias{colnames,gpu.matrix.torch-method}
\alias{dim,gpu.matrix.tensorflow-method}
\alias{dim,gpu.matrix.torch-method}
\alias{dimnames,gpu.matrix.tensorflow-method}
\alias{dimnames,gpu.matrix.torch-method}
\alias{length,gpu.matrix.tensorflow-method}
\alias{length,gpu.matrix.torch-method}
\alias{ncol,gpu.matrix.tensorflow-method}
\alias{ncol,gpu.matrix.torch-method}
\alias{nrow,gpu.matrix.tensorflow-method}
\alias{nrow,gpu.matrix.torch-method}


\alias{dim<-}
\alias{dimnames<-}

\alias{dim<--methods}
\alias{dimnames<--methods}
\alias{dim<-,gpu.matrix.tensorflow,vector-method}
\alias{dim<-,gpu.matrix.torch,vector-method}
\alias{dimnames<-,gpu.matrix.tensorflow,vector-method}
\alias{dimnames<-,gpu.matrix.torch,vector-method}



\usage{
\S4method{rownames}{gpu.matrix.tensorflow}(x)
\S4method{rownames}{gpu.matrix.torch}(x)
\S4method{colnames}{gpu.matrix.tensorflow}(x)
\S4method{colnames}{gpu.matrix.torch}(x)
\S4method{dim}{gpu.matrix.tensorflow}(x)
\S4method{dim}{gpu.matrix.torch}(x)
\S4method{dimnames}{gpu.matrix.tensorflow}(x)
\S4method{dimnames}{gpu.matrix.torch}(x)
\S4method{length}{gpu.matrix.tensorflow}(x)
\S4method{length}{gpu.matrix.torch}(x)
\S4method{ncol}{gpu.matrix.tensorflow}(x)
\S4method{ncol}{gpu.matrix.torch}(x)
\S4method{nrow}{gpu.matrix.tensorflow}(x)
\S4method{nrow}{gpu.matrix.torch}(x)

\S4method{dim}{gpu.matrix.tensorflow,vector}(x) <- value
\S4method{dim}{gpu.matrix.torch,vector}(x) <- value
\S4method{dimnames}{gpu.matrix.tensorflow,vector}(x) <- value
\S4method{dimnames}{gpu.matrix.torch,vector}(x) <- value

}


\arguments{
\item{x}{a \code{\linkS4class{gpu.matrix}}.}
\item{value}{For \code{dim} a numeric vector of length 2 with the number of rows and number of columns. For \code{dimnames} a character or numeric vector of length 2 with the names of the rows and names of the columns. }
}


\value{

\code{rownames} returns the names of the rows of a gpu.matrix-class object.
\code{colnames} returns the names of the columns of a gpu.matrix-class object.

\code{dim} returns the number of rows and columns of a gpu.matrix-class object and
 \code{dim <- } sets the number of rows and columns of a gpu.matrix-class object.

\code{dimnames} returns the names of the rows and columns of a gpu.matrix-class object and
 \code{dimnames <- } sets the names of the rows and columns of a gpu.matrix-class object.

\code{length} returns the length (ncol*nrow) of a gpu.matrix-class object.

\code{ncol} returns the number of columns of a gpu.matrix-class object.

\code{nrow} returns the number of rows of a gpu.matrix-class object.



}



\seealso{
For more information:
\code{\link[base]{rownames}},
\code{\link[base]{colnames}},
\code{\link[base]{dim}},
\code{\link[base]{dim<-}},
\code{\link[base]{dimnames}},
\code{\link[base]{dimnames<-}},
\code{\link[base]{length}},
\code{\link[base]{ncol}},
\code{\link[base]{nrow}}.

}


\examples{
\donttest{
\dontrun{

a <- gpu.matrix(rnorm(9))

dim(a) <- c(3,3) #sets the number of rows and columns.
dim(a) #shows the number of rows and the number of columns
ncol(a) #shows the number of columns
nrow(a) #shows the number of rows
length(a) #shows the lenght of the matrix (nrow*ncol)


dimnames(a) <- list(c("r1","r2","r3"),c("c1","c2","c3")) #sets rows and column names
dimnames(a) #shows both the row and the col names

#these functions are equivalent to the following:
rownames(a) <- c("r1","r2","r3") #adds rownames to a.
colnames(a) <- c("c1","c2","c3") #adds colnames to a.
rownames(a) #shows rownames.
colnames(a) #shows colnames.

}
}
}

