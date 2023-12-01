
updateH <- function(V,W,H) {
  H <- H * (t(W) %*% V)/((t(W) %*% W) %*% H)
}

updateW <- function(V,W,H) {
  W <- W * (V %*% t(H))/(W %*% (H %*% t(H)) )
}

setNegativeZero <- function(x){
  if(min(x) < 0){
    if(class(x)[[1]] == "gpu.matrix.torch"){
      x@gm <- torch::torch_clamp(x@gm, min=0)
    }else if(class(x)[[1]] == "gpu.matrix.tensorflow"){
      x@gm <- tensorflow::tf$maximum(x@gm, 0)
    }else{
      x[(x < 0)] <- 0
    }
    warning(message="The values of Hinit must be positive. Negative values in Hinit are set to 0.")
  }


  return(x)
}
controlDimensionNMF <- function(Winit=NULL, Hinit=NULL,V,k){
  if(!((nrow(Winit) == nrow(V)) & (ncol(Winit) == k))){
    stop("The dimensions of the Winit matrix are incorrect.
               Please check that nrow(Winit) == nrow(V) and that ncol(Winit) == k.")
  }
  if(!((nrow(Hinit) == k) & (ncol(Hinit) == ncol(V)))){
    stop("The dimensions of the Hinit matrix are incorrect.
               Please check that nrow(Hinit) == k and that ncol(Hinit) == ncol(V).")
  }
}
NMFgpumatrix <- function(V,k=10,Winit=NULL, Hinit=NULL, tol=1e-6, niter=100){
  set.seed(123)
  objectClass <- class(V)[[1]]
  objectPackage <- attr(class(V),"package")
  if(!is.null(objectPackage)){
    if(objectClass == "gpu.matrix.torch" | objectClass == "gpu.matrix.tensorflow"){
      if(is.null(Winit)){
        Winit <- gpu.matrix(runif(nrow(V)*k),nrow(V),k, dtype = dtype(V),type = typeGPUmatrix(V),device = device(V))
      }
      if(is.null(Hinit)){
        Hinit <- gpu.matrix(runif(k*ncol(V)),k,ncol(V), dtype = dtype(V),type = typeGPUmatrix(V),device = device(V))
      }
    }else{
      if(is.null(Winit)){
        Winit <- matrix(runif(nrow(V)*k),nrow(V),k)
      }
      if(is.null(Hinit)){
        Hinit <- matrix(runif(k*ncol(V)),k,ncol(V))
      }
    }
  }else{
    if(is.null(Winit)){
      Winit <- matrix(runif(nrow(V)*k),nrow(V),k)
    }
    if(is.null(Hinit)){
      Hinit <- matrix(runif(k*ncol(V)),k,ncol(V))
    }
  }


  Winit <- setNegativeZero(Winit)
  Hinit <- setNegativeZero(Hinit)
  controlDimensionNMF(Winit, Hinit,V,k)
  V <- setNegativeZero(V)




  Vold <- V
  condition <- F
  for (iter in 1:niter) {
    Winit <- updateW(V,Winit,Hinit)
    Hinit <- updateH(V,Winit,Hinit)
    Vnew <- Winit%*%Hinit
    if(mean((Vnew-Vold)^2)<tol){
      res <- list("W"=Winit,"H"=Hinit)
      warning(message="Early finish")
      return(res)
      break()
    }
    Vold <- Vnew
    if(iter == niter){
      res <- list("W"=Winit,"H"=Hinit)
      return(res)
    }
  }

  # if(!condition){
  #
  #   warning(message="Early finish")
  # }
  # return(res)
}



sigmoid <- function(x) {
  1/(1+exp(-x))
}

LR_GradientConjugate_gpumatrix <- function(X,y,beta = NULL, lambda = 0, iterations = 100, tol = 1e-8) {
  tX <- t(X)
  if (is.null(beta)){
    objectPackage <- attr(class(X),"package")
    if(!is.null(objectPackage)){
      if (objectPackage == "GPUmatrix"){
        beta <- gpu.matrix(0,nrow = ncol(X),ncol=1, device = device(X), dtype=dtype(X))
      }else{
        beta <- rep(0,ncol(X))
      }

    }else{
      beta <- rep(0,ncol(X))
    }
  }

  p <- sigmoid(X %*% beta)
  a <- p*(1-p)
  g <- tX %*% (p - y)
  u <- g
  devold <- 0
  for (iter in 1:iterations) {
    if (iter == 1) {
      uhu <- (sum(u*(tX %*% (a * (X %*% u)))) + lambda * sum(u*u))
      beta <- beta-sum(g*u)/uhu * u
    } else{
      p <- sigmoid(X %*% beta)

      g_old <- g
      a <- p*(1-p)
      g <- tX %*% (p - y)
      k <- g - g_old
      beta_coef <- sum(g * k)/sum(u*k) # Hestenes-Stiefel update. Other options are possible
      u <- g - u * beta_coef

      uhu <- sum((X %*% u)^2*a) + lambda * sum(u*u) +1e-10
      beta <- beta - sum(g*u)/uhu * u
      dev <- dev.resids(y, p)
      if (abs(dev - devold)/(0.1 + abs(dev)) < tol)
        break
      devold <- dev
    }
  }
  return(beta)
}

dev.resids <- function(y, p, epsilon = 1e-9) {

  p_adj <- (p + epsilon)/ (1+2*epsilon)
  s <- 2*(-sum(y*log(p_adj)+(1-y)*log(1-p_adj)))
  return(s)
}


GPUglm <- function(...) {
  res <- glm(..., method = "glm.fit.GPU")
  class(res) <- "GPUglm"

  return(res)
}


glm.fit.GPU <- function (x,y, intercept = TRUE, weights = NULL,
                         family = gaussian(), start = NULL, etastart = NULL, mustart = NULL,
                         offset = NULL, acc = 1e-08, maxit = 25, k = 2,
                         sparse = NULL,
                         trace = FALSE,
                         dtype="float64",
                         device=NULL,
                         type=NULL,...)
{

  method = c("normal")
  nobs <- NROW(y)
  nvar <- ncol(x)

  if (missing(y))
    stop("Argument y is missing")
  if (missing(x))
    stop("Argument x is missing")
  if (is.null(offset))
    offset <- rep.int(0, nobs)
  if (is.null(weights))
    weights <- rep(1, nobs)

  col.names <- dimnames(x)[[2]]

  fam <- family$family
  link <- family$link
  variance <- family$variance
  dev.resids <- family$dev.resids
  aic <- family$aic
  linkinv <- family$linkinv
  mu.eta <- family$mu.eta


  if (is.null(sparse))
    sparse <- F
  if (is.null(start)) {
    if (is.null(mustart))
      eval(family$initialize)
    eta <- if (is.null(etastart))
      family$linkfun(mustart)
    else etastart
    mu <- mustart
    start <- rep(0, nvar)
  }
  else {
    eta <- offset + as.vector(if (nvar == 1)
      x * start
      else {
        if (sparse)
          x %*% start
        else tcrossprod(x, t(start))
      })
    mu <- linkinv(eta)
  }
  iter <- 0
  dev <- sum(dev.resids(y, c(mu), weights))

  #GPUmatrix initialization
  x <- gpu.matrix(x,dtype = dtype, type = type, device = device)
  y <- gpu.matrix(y,dtype = dtype, type = type, device = device)
  tol <- 1

  if ((fam == "gaussian") & (link == "identity"))
    maxit <- 1

  while ((tol > acc) & (iter < maxit)) {
    iter <- iter + 1
    beta <- start
    dev0 <- dev
    varmu <- variance(mu)
    mu.eta.val <- mu.eta(as.numeric(eta))
    z <- (eta - offset) + (y - mu)/mu.eta.val
    W <- gpu.matrix((weights * mu.eta.val * mu.eta.val)/varmu,
                    dtype = dtype, type = type, device = device )
    XTX <- crossprod(x*W, x)
    XTz <- t(crossprod((W * z), x))
    variable <- colnames(x)

    ris <- list(rank = nvar, pivot = 1:nvar)
    ok <- ris$pivot[1:ris$rank]

    beta <- start

    start <- chol_solve(t(chol(XTX)), XTz)

    eta <- drop(x %*% start)


    mu <- linkinv(as.numeric(eta <- eta + offset))
    dev <- sum(dev.resids(as.numeric(y), c(mu), weights))
    tol <- max(abs(dev0 - dev)/(abs(dev) + 0.1))
    if (trace)
      cat("iter", iter, "tol", tol, "\n")
  }

  wt <- sum(weights)
  wtdmu <- if (intercept)
    as.numeric(sum(weights * y)/wt)
  else linkinv(offset)
  nulldev <- sum(dev.resids(as.numeric(y), c(wtdmu), weights))
  n.ok <- nobs - sum(weights == 0)
  nulldf <- n.ok - as.integer(intercept)
  rank <- ris$rank
  dfr <- nobs - rank - sum(weights == 0)
  aic.model <- aic(as.numeric(y), nobs, as.numeric(mu), weights, dev) + k * rank
  ll.nuovo <- ll.speedglmGPU(fam, aic.model, rank)
  res <- (y - mu)/mu.eta(as.numeric(eta))
  resdf <- n.ok - rank
  RSS <- sum(W * res * res)
  var_res <- RSS/dfr
  dispersion <- if (fam %in% c("poisson", "binomial"))
    1
  else var_res

  coefficients <- rep(NA, nvar)
  start <- as(start, "numeric")
  coefficients[ok] <- start

  names(coefficients) <- if (is.null(col.names) & (!is.null(coefficients))) {
    if (intercept) {
      if (length(coefficients) > 1)
        c("(Intercept)", paste("V", 1:(length(coefficients) -
                                         1), sep = ""))
      else "(Intercept)"
    }
    else paste("V", 1:length(coefficients), sep = "")
  }
  else col.names
  if ((fam == "gaussian") & (link == "identity"))
    tol <- 0
  rval <- list(coefficients = coefficients, logLik = ll.nuovo,
               iter = iter, tol = tol, family = family, link = link,
               df = dfr, XTX = as.matrix(XTX), dispersion = dispersion, ok = ok,
               rank = rank, RSS = as.numeric(RSS), method = method, aic = aic.model,
               offset = offset, sparse = sparse, deviance = dev, nulldf = nulldf,
               nulldev = nulldev, ngoodobs = n.ok, n = nobs, intercept = intercept,
               convergence = (!(tol > acc)),converged = (!(tol > acc)))

  class(rval) <- "GPUglm"
  return(rval)
}

ll.speedglmGPU <- function (family, aic.model, nvar) {
  switch(family,
         binomial = -(aic.model - 2 * nvar)/2,
         Gamma = -((aic.model - 2 * nvar) - 2)/2,
         gaussian = -((aic.model - 2 * nvar) - 2)/2,
         poisson = -(aic.model - 2 * nvar)/2,
         inverse.gaussian = -((aic.model - 2 * nvar) - 2)/2,
         quasi = NA,
         quasibinomial = NA,
         quasipoisson = NA)
}

summary.GPUglm <- function (object, correlation = FALSE, ...)
{
  if (!inherits(object, "GPUglm"))
    stop("object is not of class GPUglm")
  z <- object
  var_res <- as.numeric(z$RSS/z$df)
  dispersion <- if (z$family$family %in% c("poisson", "binomial")) 1 else var_res
  if (z$method == "qr") {
    z$XTX <- z$XTX[z$ok, z$ok]
  }
  inv <- solve(z$XTX, tol = z$tol.solve)
  covmat <- diag(inv)
  se_coef <- rep(NA, length(z$coefficients))
  se_coef[z$ok] <- sqrt(dispersion * covmat)
  if (z$family$family %in% c("binomial", "poisson")) {
    z1 <- z$coefficients/se_coef
    p <- 2 * pnorm(abs(z1), lower.tail = FALSE)
  } else {
    t1 <- z$coefficients/se_coef
    p <- 2 * pt(abs(t1), df = z$df, lower.tail = FALSE)
  }
  dn <- c("Estimate", "Std. Error")
  if (z$family$family %in% c("binomial", "poisson")) {

    param <- data.frame(z$coefficients, se_coef, z1,p)
    dimnames(param) <- list(names(z$coefficients), c(dn,
                                                     "z value", "Pr(>|z|)"))
  } else {
    param <- data.frame(z$coefficients, se_coef, t1,p)
    dimnames(param) <- list(names(z$coefficients), c(dn,
                                                     "t value", "Pr(>|t|)"))
  }
  eps <- 10 * .Machine$double.eps
  if (z$family$family == "binomial") {
    if (any(z$mu > 1 - eps) || any(z$mu < eps))
      warning("fitted probabilities numerically 0 or 1 occurred")
  }
  if (z$family$family == "poisson") {
    if (any(z$mu < eps))
      warning("fitted rates numerically 0 occurred")
  }
  keep <- match(c("call", "terms", "family", "deviance", "aic",
                  "df", "nulldev", "nulldf", "iter", "tol", "n", "convergence",
                  "ngoodobs", "logLik", "RSS", "rank"), names(object),
                0)
  ans <- c(object[keep], list(coefficients = param, dispersion = dispersion,
                              correlation = correlation, cov.unscaled = inv, cov.scaled = inv *
                                var_res))
  if (correlation) {
    ans$correl <- (inv * var_res)/outer(na.omit(se_coef),
                                        na.omit(se_coef))
  }
  class(ans) <- "summary.GPUglm"
  return(ans)
}


print.summary.GPUglm <- function (x, digits = max(3, getOption("digits") - 3), ...)
{
  cat("Generalized Linear Model of class 'summary.GPUglm':\n")
  if (!is.null(x$call))
    cat("\nCall: ", deparse(x$call), "\n\n")
  if (length(x$coef)) {
    cat("Coefficients:\n")
    cat(" ------------------------------------------------------------------",
        "\n")
    sig <- function(z){
      if (!is.na(z)){
        if (z < 0.001)
          "***"
        else if (z < 0.01)
          "** "
        else if (z < 0.05)
          "*  "
        else if (z < 0.1)
          ".  "
        else "   "
      } else "   "
    }
    sig.1 <- sapply(as.numeric(as.character(x$coefficients[,4])),
                    sig)
    est.1 <- cbind(format(x$coefficients, digits = digits),
                   sig.1)
    colnames(est.1)[ncol(est.1)] <- ""
    print(est.1)
    cat("\n")
    cat("-------------------------------------------------------------------",
        "\n")
    cat("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1",
        "\n")
    cat("\n")
  }
  else cat("No coefficients\n")
  cat("---\n")
  cat("null df: ", x$nulldf, "; null deviance: ", round(x$nulldev,
                                                        digits = 2), ";\n", "residuals df: ", x$df, "; residuals deviance: ",
      round(x$deviance, digits = 2), ";\n", "# obs.: ", x$n,
      "; # non-zero weighted obs.: ", x$ngoodobs, ";\n", "AIC: ",
      x$aic, "; log Likelihood: ", x$logLik, ";\n", "RSS: ",
      round(x$RSS, digits = 1), "; dispersion: ", x$dispersion,
      "; iterations: ", x$iter, ";\n", "rank: ", round(x$rank,
                                                       digits = 1), "; max tolerance: ", format(x$tol, scientific = TRUE,
                                                                                                digits = 3), "; convergence: ", x$convergence, ".\n",
      sep = "")
  invisible(x)
  if (x$correlation) {
    cat("---\n")
    cat("Correlation of Coefficients:\n")
    x$correl[upper.tri(x$correl, diag = TRUE)] <- NA
    print(x$correl[-1, -nrow(x$correl)], na.print = "", digits = 2)
  }
}


print.GPUglm <- function(x,digits = max(3, getOption("digits") - 3),...)
{
  cat("Generalized Linear Model of class 'gpu.matrix':\n")
  if (!is.null(x$call)) cat("\nCall: ", deparse(x$call), "\n\n")
  if (length(x$coef)) {
    cat("Coefficients:\n")
    print.default(format(x$coefficients, digits = digits), print.gap = 2,
                  quote = FALSE)
  } else cat("No coefficients\n")
  cat("\n")
  invisible(x)
}
