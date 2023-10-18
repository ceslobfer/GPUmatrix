# setMethod("Summary",
#           signature = c(x="gpu.matrix.torch"),
#           function(x)
#           {
#             op = .Generic[[1]]
#             switch(op,
#                    'sum' = {
#                      if (x@sparse) {
#                        res <- as.numeric(torch::torch_sum(x@gm$values())$cpu())
#                      }else{
#                        res <- as.numeric(x@gm$sum()$cpu())
#                      }
#                    },
#                    'min' = {
#                      if(x@sparse){
#                        res <- as.numeric(tensorflow::tf$reduce_min(x@gm$values))
#                      } else{
#                        res <- as.numeric(tensorflow::tf$reduce_min(x@gm))
#                      }
#                    },
#                    'max'={
#                      if (x@sparse) {
#                        res <- as.numeric(tensorflow::tf$sparse$reduce_max(x@gm))
#                      }else{
#                        res <-as.numeric(tensorflow::tf$reduce_max(x@gm))
#                      }
#                    }
#             )
#             return(res)
#           }
# )
