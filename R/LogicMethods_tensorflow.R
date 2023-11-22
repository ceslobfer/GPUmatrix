setMethod("Logic",
          c(e1="gpu.matrix.tensorflow", e2="ANY"),
          function(e1, e2)
          {
            op = .Generic[[1]]
            castMatrix <- castTypeOperations(e1,e2)
            e1 <- castMatrix[[1]]
            e2 <- castMatrix[[2]]
            dtype(e2) <- "bool"
            dtype(e2) <- "bool"

            switch(op,
                   '&' = {
                     return(gpu.matrix.tensorflow(e1@gm&e2@gm))
                   },
                   '|' = {
                     return(gpu.matrix.tensorflow(e1@gm|e2@gm))
                   }
            )
          }
)

setMethod("Logic",
          c(e1="ANY", e2="gpu.matrix.tensorflow"),
          function(e1, e2)
          {
            op = .Generic[[1]]
            castMatrix <- castTypeOperations(e1,e2)
            e1 <- castMatrix[[1]]
            e2 <- castMatrix[[2]]
            dtype(e2) <- "bool"
            dtype(e2) <- "bool"

            switch(op,
                   '&' = {
                     return(gpu.matrix.tensorflow(e1@gm&e2@gm))
                   },
                   '|' = {
                     return(gpu.matrix.tensorflow(e1@gm|e2@gm))
                   }
            )
          }
)
