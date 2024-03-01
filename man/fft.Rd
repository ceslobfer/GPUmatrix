\name{fft}
\title{Fast Discrete Fourier Transform (FFT)}
\description{
The function \code{fft}  mimics the function \code{fft} of the library 'stats' to compute on gpu.matrix-class objects: it "Computes the Discrete Fourier Transform (DFT) of an array with a fast algorithm, the 'Fast Fourier Transform' (FFT)."

The function \code{mvfft} mimics the function \code{mvfft} of the library 'stats' which: "takes a real or complex matrix as argument, and returns a similar shaped matrix, but with each column replaced by its discrete Fourier transform".

}

\alias{fft}
\alias{fft-methods}
\alias{fft,gpu.matrix.tensorflow-method}
\alias{fft,gpu.matrix.tensorflow,missing-method}
\alias{fft,gpu.matrix.torch-method}
\alias{fft,gpu.matrix.torch,logical-method}
\alias{fft,gpu.matrix.torch,missing-method}

\alias{mvfft}
\alias{mvfft-methods}
\alias{mvfft,gpu.matrix.torch-method}
\alias{mvfft,gpu.matrix.torch,logical-method}
\alias{mvfft,gpu.matrix.torch,missing-method}
\alias{mvfft,gpu.matrix.tensorflow-method}
\alias{mvfft,gpu.matrix.tensorflow,missing-method}


\usage{
\S4method{fft}{gpu.matrix.tensorflow}(z)
\S4method{fft}{gpu.matrix.torch}(z)
\S4method{fft}{gpu.matrix.torch,logical}(z,inverse)

\S4method{mvfft}{gpu.matrix.torch}(z)
\S4method{mvfft}{gpu.matrix.tensorflow}(z)
\S4method{mvfft}{gpu.matrix.torch,logical}(z,inverse)
}

\arguments{
\item{z}{a  \code{\linkS4class{gpu.matrix}} object containing the values to be transformed.}
\item{inverse}{the same as in the library 'stats': "if TRUE, the unnormalized inverse transform is computed (the inverse has a +in the exponent of \eqn{e}, but here, we do not divide by \code{1/length(x)})". By default is FALSE. Plea Note that this parameter only work for torch.}
}

\seealso{
For more information see:
\code{\link[stats]{fft}}, \code{\link[torch]{torch_fft_ifft}}, and \code{\link[torch]{torch_fft_fft}}.
}

\details{
The function \code{fft} mimics the function \code{\link[stats]{fft}} to operate on gpu.matrix-class objects of one dimension. If the input gpu.matrix \code{z} has tow dimensions the function will not work, as the method for two dimensions is not implemented yet for gpu.matrix-class objects. In this case the function will display the following error message: "FFT in gpu.matrix with 2 dimensions is not allowed yet".

The function \code{mvfft} mimics the function \code{\link[stats]{mvfft}} to operate on gpu.matrix-class objects. This function will apply the discrete Fourier transform to each column of the input \code{z} matrix.

Note that the \code{inverse} parameter only works for 'torch' for both \code{fft} and \code{mvfft} functions.


The functions \code{fft} and \code{mvfft} internally call the corresponding function of the library torch or tensorflow (depending on the type of input gpu.matrix-class).

If the input gpu.matrix-class object(s) are stored on the GPU, then the operations will be performed on the GPU. See \code{\link{gpu.matrix}}.

}

\value{
It returns a gpu.matrix-class object with the transformed values. To access the real and imaginary information use the function \code{Re()} for teh rea part and \code{Im()} for the imaginary part. Furthermore, the following code can be used:
\code{output@gm$real} for the real part and \code{output@gm$imag} for the imaginary part.
}


\examples{
if(installTorch()){

x <- gpu.matrix(1:4,ncol = 1)
output_gpu <- fft(x)
output_matrix <- fft(z = as.matrix(x))

#check results:
Re(output_gpu)
Re(output_matrix)
Im(output_gpu)
Im(output_matrix)


x <- gpu.matrix(1:12,ncol = 3)
output_gpu <- mvfft(x)
output_matrix <- mvfft(as.matrix(x))

#check results:
Re(output_gpu)
Re(output_matrix)
Im(output_gpu)
Im(output_matrix)
}


}



