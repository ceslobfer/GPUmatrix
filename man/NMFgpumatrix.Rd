\name{NMFgpumatrix}
\title{Non negative factorization of a matrix}

\description{
The non-negative factorization (NMF) of a matrix is an approximate factorization were an initial matrix \code{V} is approximated by the product of two matrix \code{W} and \code{H} so that,

\eqn{V \approx WH}

This function operates in the same way with the 'base' \code{matrix} objects as with gpu.matrix-class objects, and it does not require any additional changes beyond initializing the input matrix as a gpu.matrix-class object.

}

\alias{NMFgpumatrix}

\usage{
NMFgpumatrix(V, k = 10, Winit = NULL,
             Hinit = NULL, tol = 1e-06,
             niter = 100)
}
\arguments{
  \item{V}{a \code{\linkS4class{gpu.matrix}}.}
  \item{k}{The inner dimension of the product of the matrices W and H. That is, it corresponds to the number of columns in W and the number of rows in H.}
  \item{Winit}{Initial value for matrix W.}
  \item{Hinit}{Initial value for matrix H.}
  \item{tol}{tolerance to be used for the estimation.}
  \item{niter}{maximum number of iterations.}
}

\details{
We have implemented our own non-negative matrix factorization (NMF) function using Lee
and Seung[1] multiplicative update rule:

\eqn{W_{[i,j]}^{n+1} \leftarrow  W_{[i,j]}^{n} \frac{(V(H^{n+1})^T)_{[i,j]}}{(W^nH^{n+1}(H^{n+1})^T)_{[i,j]}}}

and

\eqn{H_{[i,j]}^{n+1} \leftarrow  H_{[i,j]}^{n} \frac{((W^{n})^TV)_{[i,j]}}{((W^n)^TH^{n}H^{n})_{[i,j]}}}

to update the \eqn{W} and \eqn{H} respectively.

If the input gpu.matrix-class object is stored on the GPU, then the operations will be performed on the GPU. See \code{\link{gpu.matrix}}.


}

\references{
[1] Lee, D., Seung, H. Learning the parts of objects by non-negative matrix factorization. Nature 401, 788–791 (1999). https://doi.org/10.1038/44565
}

\author{
Angel Rubio and Cesar Lobato.
}


\value{
The function returns a list that contains the corresponding matrix \eqn{W} and \eqn{H}. If the input \code{V} matrix is a gpu.matrix-class object, then both \eqn{W} and \eqn{H} are also gpu.matrix-class objects.
}
\examples{
\donttest{
\dontrun{
#toy exapmle
a1 <- gpu.matrix(rnorm(9),nrow=3,ncol=3)
a2 <- gpu.matrix(rnorm(9),nrow=3,ncol=3)
V <- a1 \%*\% a2
b <- NMFgpumatrix(V = V)
#check result:
V - b$W \%*\% b$H

}
}
}

