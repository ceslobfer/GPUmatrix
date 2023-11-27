setMethod("Complex",
          c(z="gpu.matrix.tensorflow"),
          function(z)
          {
            op = .Generic[[1]]
            switch(op,
                   'Re' = {
                     z@gm <- tensorflow::tf$math$real(z@gm)
                   },
                   'Im' = {
                     z@gm <- tensorflow::tf$math$imag(z@gm)
                   },
                   'Conj' = {
                     z@gm <- tensorflow::tf$math$conj(z@gm)
                   },
                   'Arg' = {
                     z@gm <- tensorflow::tf$math$angle(z@gm)
                   },
                   'Mod'={
                     z@gm <- tensorflow::tf$math$abs(z@gm)
                   }
            )
            return(z)
          }
)
