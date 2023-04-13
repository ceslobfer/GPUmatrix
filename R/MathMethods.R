setMethod("Math",
          signature(x="gpu.matrix.tensorflow"),
          function(x)
          {
            x <- warningInteger(x)
            op = .Generic[[1]]
            switch(op,
                   'log' = {
                     x <- warningSparseTensor(x)
                     x@gm <- tensorflow::tf$math$log(x@gm)
                     return(x)
                   },
                   'log2' = {
                     x <- warningSparseTensor(x)
                     x@gm <- tensorflow::tf$experimental$numpy$log2(x@gm)
                     return(x)
                   },
                   'log10' = {
                     x <- warningSparseTensor(x)
                     x@gm <- tensorflow::tf$experimental$numpy$log10(x@gm)
                     return(x)
                   },
                   'log1p' = {
                     x <- warningSparseTensor(x)
                     x@gm <- tensorflow::tf$math$log1p(x@gm)
                     return(x)
                   },
                   'cos' = {
                     x <- warningSparseTensor(x)
                     x@gm <- tensorflow::tf$math$cos(x@gm)
                     return(x)
                   },
                   'cosh' = {
                     x <- warningSparseTensor(x)
                     x@gm <- tensorflow::tf$math$cosh(x@gm)
                     return(x)
                   },
                   'acos' = {
                     x <- warningSparseTensor(x)
                     x@gm <- tensorflow::tf$math$acos(x@gm)
                     return(x)
                   },
                   'acosh' = {
                     x <- warningSparseTensor(x)
                     x@gm <- tensorflow::tf$math$acosh(x@gm)
                     return(x)
                   },
                   'sin' = {
                     x <- warningSparseTensor(x)
                     x@gm <- tensorflow::tf$math$sin(x@gm)
                     return(x)
                   },
                   'sinh' = {
                     x <- warningSparseTensor(x)
                     x@gm <- tensorflow::tf$math$sinh(x@gm)
                     return(x)
                   },
                   'asin' = {
                     x <- warningSparseTensor(x)
                     x@gm <- tensorflow::tf$math$asin(x@gm)
                     return(x)
                   },
                   'asinh' = {
                     x <- warningSparseTensor(x)
                     x@gm <- tensorflow::tf$math$asinh(x@gm)
                     return(x)
                   },
                   'tan' = {
                     x <- warningSparseTensor(x)
                     x@gm <- tensorflow::tf$math$tan(x@gm)
                     return(x)
                   },
                   'atan' = {
                     x <- warningSparseTensor(x)
                     x@gm <- tensorflow::tf$math$atan(x@gm)
                     return(x)
                   },
                   'tanh' = {
                     x <- warningSparseTensor(x)
                     x@gm <- tensorflow::tf$math$tanh(x@gm)
                     return(x)
                   },
                   'atanh' = {
                     x <- warningSparseTensor(x)
                     x@gm <- tensorflow::tf$math$atanh(x@gm)
                     return(x)
                   },
                   'sqrt' = {
                     x@gm <- tensorflow::tf$math$sqrt(x@gm)
                     return(x)
                   },
                   'abs' = {
                     x@gm <- tensorflow::tf$abs(x@gm)
                     return(x)
                   },
                   'sign' = {
                     x@gm <- tensorflow::tf$math$sign(x@gm)
                     return(x)
                   },
                   'ceiling' = {
                     x <- warningSparseTensor(x)
                     x@gm <- tensorflow::tf$math$ceil(x@gm)
                     return(x)
                   },
                   'floor' = {
                     x <- warningSparseTensor(x)
                     x@gm <- tensorflow::tf$math$floor(x@gm)
                     return(x)
                   },
                   'cumsum' = {
                     x <- warningSparseTensor(x)
                     res <- as.vector(tensorflow::tf$math$cumsum(x@gm))
                     return(res)
                   },
                   'cumprod' = {
                     x <- warningSparseTensor(x)
                     res <- as.vector(tensorflow::tf$math$cumprod(x@gm))
                     return(res)
                   },
                   'exp' = {
                     x <- warningSparseTensor(x)
                     x@gm <- tensorflow::tf$math$exp(x@gm)
                     return(x)
                   },
                   'expm1' = {
                     x <- warningSparseTensor(x)
                     x@gm <- tensorflow::tf$math$expm1(x@gm)
                     return(x)
                   }
            )
          }
)
