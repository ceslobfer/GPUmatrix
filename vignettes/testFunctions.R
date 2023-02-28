
a <- matrix(c(5,1,1,3),2,2)
a <- matrix(rnorm(16),4,4)

x <- gpu.matrix(a, type = "tensorflow")
xS <- gpu.matrix(a, type = "tensorflow",sparse = T)
y <- gpu.matrix(a)
yS <- gpu.matrix(a,sparse = T)


determinant(y)
determinant(yS)
determinant(x)
determinant(xS)
determinant(x,logarithm = F)
determinant(xS,logarithm = F)
determinant(y,logarithm = F)
determinant(yS,logarithm = F)

det(x)
det(xS)
det(y)
det(yS)

fft(x)
fft(xS)
fft(y)
fft(yS)

sort(x)
sort(xS)
sort(y)
sort(yS)

sort(x,decreasing = T)
sort(xS,decreasing = T)
sort(y,decreasing = T)
sort(yS,decreasing = T)

round(x)
round(xS)
round(y)
round(yS)

round(x,digits = 1)
round(xS,digits = 1)
round(y,digits = 1)
round(yS,digits = 1)

length(x)
length(xS)
length(y)
length(yS)

dim(x)
dim(xS)
dim(y)
dim(yS)

dim(a)<- c(2,8)
# a
dim(x)<- c(2,8)
# x
dim(xS)<- c(2,8)
# to_dense(xS)
dim(y)<- c(2,8)
# y
dim(yS)<- c(2,8)
# to_dense(yS)

rownames(x) <- c("a","b")
rownames(xS) <- c("a","b")
rownames(y) <- c("a","b")
rownames(yS) <- c("a","b")

dimnames(x) <- list(c("b","b"),c("a","b","c","d","e","f","g","h"))
dimnames(a) <- list(c("b","b"),c("a","b","c","d","e","f","g","h"))
dimnames(xS)<- list(c("b","b"),c("a","b","c","d","e","f","g","h"))
dimnames(y)<- list(c("b","b"),c("a","b","c","d","e","f","g","h"))
dimnames(yS)<- list(c("b","b"),c("a","b","c","d","e","f","g","h"))

# dimnames(x)
# dimnames(xS)
# dimnames(y)
# dimnames(yS)

x[,1]
xS[,1]
y[,1]
yS[,1]
x[,c(1,2)]
xS[,c(1,2)]
y[,c(1,2)]
yS[,c(1,2)]
x[1,]
xS[1,]
y[1,]
yS[1,]
x[c(1,2),]
xS[c(1,2),]
y[c(1,2),]
yS[c(1,2),]
a[16]
x[16]
xS[16]
y[16]
yS[16]
x[c(1,16)]
xS[c(1,16)]
y[c(1,16)]
yS[c(1,16)]
x[c(1,1),c(1,2)]
to_dense(xS[c(1,1),c(1,2)])
y[c(1,1),c(1,2)]
to_dense(yS[c(1,1),c(1,2)])


rowSums(x)
rowSums(xS)
rowSums(y)
rowSums(yS)

colSums(x)
colSums(xS)
colSums(y)
colSums(yS)


cbind(x,c(1,2))
to_dense(cbind(xS,c(1,2)))
cbind(y,c(1,2))
to_dense(cbind(yS,c(1,2)))

rbind(x,c(1,2,3,4,1,2,3,4))
to_dense(rbind(xS,c(1,2,3,4,1,2,3,4)))
rbind(y,c(1,2,3,4,1,2,3,4))
to_dense(rbind(yS,c(1,2,3,4,1,2,3,4)))

cbind(x,x)
to_dense(cbind(xS,xS))
cbind(x,xS)
cbind(xS,a)
cbind(x,a)
cbind(M,x)
cbind(y,y)
to_dense(cbind(yS,yS))
cbind(y,yS)
cbind(yS,a)
cbind(y,a)
cbind(M,y)

rbind(x,x)
to_dense(rbind(xS,xS))
rbind(x,xS)
rbind(xS,a)
rbind(x,a)
rbind(M,x)
rbind(y,y)
to_dense(rbind(yS,yS))
rbind(y,yS)
rbind(yS,a)
rbind(y,a)
rbind(M,y)

rbind(x,c(1,2,3,4,1,2,3,4))
to_dense(rbind(xS,c(1,2,3,4,1,2,3,4)))
rbind(y,c(1,2,3,4,1,2,3,4))
to_dense(rbind(yS,c(1,2,3,4,1,2,3,4)))

head(x)
head(xS)
head(y)
head(yS)
head(x,1)
head(xS,1)
head(y,1)
head(yS,1)

tail(x)
tail(xS)
tail(y)
tail(yS)
tail(x,1)
tail(xS,1)
tail(y,1)
tail(yS,1)

nrow(x)
nrow(xS)
nrow(y)
nrow(yS)
ncol(x)
ncol(xS)
ncol(y)
ncol(yS)

t(x)
to_dense(t(xS))
t(y)
to_dense(t(yS))

library(Matrix)
M <- Matrix(a)
crossprod(x)
crossprod(xS)
crossprod(y)
crossprod(yS)
crossprod(x,a)
crossprod(xS,a)
crossprod(y,a)
crossprod(yS,a)
crossprod(x,M)
crossprod(xS,M)
crossprod(y,M)
crossprod(yS,M)

tcrossprod(x)
tcrossprod(xS)
tcrossprod(y)
tcrossprod(yS)
tcrossprod(x,a)
tcrossprod(xS,a)
tcrossprod(y,a)
tcrossprod(yS,a)
tcrossprod(x,M)
tcrossprod(xS,M)
tcrossprod(y,M)
tcrossprod(yS,M)

x %x% x
x %x% xS
x %x% a
x %x% M
xS %x% x
xS %x% xS
xS %x% a
xS %x% M
y %x% y
y %x% yS
y %x% a
y %x% M
yS %x% y
yS %x% yS
yS %x% a
yS %x% M

x + x
x + xS
x + a
x + M
xS + x
xS + xS
xS + a
xS + M
y + y
y + yS
y + a
y + M
yS + y
yS + yS
yS + a
yS + M

x * x
x * xS
x * a
x * M
xS * x
xS * xS
xS * a
xS * M
y * y
y * yS
y * a
y * M
yS * y
yS * yS
yS * a
yS * M

x / x
x / xS
x / a
x / M
xS / x
xS / xS
xS / a
xS / M
y / y
y / yS
y / a
y / M
yS / y
yS / yS
yS / a
yS / M

x + c(1,2)
x + c(1,2)
x + c(1,2)
x + c(1,2)
xS + c(1,2)
xS + c(1,2)
xS + c(1,2)
xS + c(1,2)
y + c(1,2)
y + c(1,2)
y + c(1,2)
y + c(1,2)
yS + c(1,2)
yS + c(1,2)
yS + c(1,2)
yS + c(1,2)

x %^% 2
xS %^% 2
y %^% 2
yS %^% 2

diag(x)
diag(xS)
diag(y)
diag(yS)

diag(x) <- c(1,2,3,4)
diag(xS) <- c(1,2,3,4)
diag(y) <- c(1,2,3,4)
diag(yS) <- c(1,2,3,4)

solve(x)
solve(xS)
solve(y)
solve(yS)

solve(x,t(x))
solve(xS,t(xS))
solve(y,t(y))
solve(yS,t(yS))

qr(x)
qr(xS)
qr(y)
qr(yS)


rankMatrix(x)
rankMatrix(xS)
rankMatrix(y)
rankMatrix(yS)

eigen(x)
eigen(xS)
eigen(y)
eigen(yS)

svd(x)
svd(xS)
svd(y)
svd(yS)

ginv(x)
ginv(xS)
ginv(y)
ginv(yS)

chol(a)
chol(x)
chol(xS)
chol(y)
chol(yS)

chol_solve(a,a)
chol_solve(x,x)
chol_solve(xS,x)
chol_solve(xS,xS)
chol_solve(xS,a)
chol_solve(y,y)
chol_solve(yS,y)
chol_solve(yS,yS)
chol_solve(yS,a)


mean(x)
mean(xS)
mean(y)
mean(yS)

density(x)
density(xS)
density(y)
density(yS)

hist(x)
hist(xS)
hist(y)
hist(yS)

colMeans(x)
colMeans(xS)
colMeans(y)
colMeans(yS)

rowMeans(x)
rowMeans(xS)
rowMeans(y)
rowMeans(yS)

sum(x)
sum(xS)
sum(y)
sum(yS)

dtype(x)
dtype(xS)
dtype(y)
dtype(yS)

dtype(x) <- "float32"
dtype(xS) <- "float32"
dtype(y) <- "float32"
dtype(yS) <- "float32"


min(x)
min(xS)
min(y)
min(yS)

max(x)
max(xS)
max(y)
max(yS)

which.max(x)
which.max(xS)
which.max(y)
which.max(yS)

which.min(x)
which.min(xS)
which.min(y)
which.min(yS)

cov(x)
cov(xS)
cov(y)
cov(yS)

cov2cor(x)
cov2cor(xS)
cov2cor(y)
cov2cor(yS)

cor(x)
cor(xS)
cor(y)
cor(yS)

cor(a,a)
cor(x,x)
cor(xS,x)
cor(xS,xS)
cor(xS,a)
cor(y,y)
cor(yS,y)
cor(yS,yS)
cor(yS,a)



cor(a,a,method = "spearman")
cor(x,x,method = "spearman")
cor(xS,x,method = "spearman")
cor(xS,xS,method = "spearman")
cor(xS,a,method = "spearman")
cor(y,y,method = "spearman")
cor(yS,y,method = "spearman")
cor(yS,yS,method = "spearman")
cor(yS,a,method = "spearman")

cor(a,method = "spearman")
cor(x,method = "spearman")
cor(xS,method = "spearman")
cor(xS,method = "spearman")
cor(xS,method = "spearman")
cor(y,method = "spearman")
cor(yS,method = "spearman")
cor(yS,method = "spearman")
cor(yS,method = "spearman")


rowVars(x)
rowVars(xS)
rowVars(y)
rowVars(yS)

colVars(x)
colVars(xS)
colVars(y)
colVars(yS)

rowMaxs(x)
rowMaxs(xS)
rowMaxs(y)
rowMaxs(yS)
colMaxs(x)
colMaxs(xS)
colMaxs(y)
colMaxs(yS)


rowMins(x)
rowMins(xS)
rowMins(y)
rowMins(yS)
colMins(x)
colMins(xS)
colMins(y)
colMins(yS)

rowRanks(x)
rowRanks(xS)
rowRanks(y)
rowRanks(yS)
colRanks(x)
colRanks(xS)
colRanks(y)
colRanks(yS)

x==x
xS == x
y==y
y==yS
yS==y
x>(2*x)
xS>(2*x)
y>(2*y)
y>(2*yS)
yS>(2*y)


# Ejemplo de aplicación de cada función de la lista a cada una de las 4 matrices

# Definimos las matrices
x <- gpu.matrix(1:6, nrow = 2, type = "tensorflow", dtype = "float64")
xS <- gpu.matrix(c(1,-1,2,-2,3,-3), nrow = 2, sparse=T, type = "tensorflow")
y <- gpu.matrix(c(0, pi/4, pi/2, 3*pi/4, pi, 5*pi/4, 3*pi/2, 7*pi/4), nrow = 2)
yS <- gpu.matrix(c(0.5, 1, 2, 3, 4, 5, 6, 7), nrow = 2)

# Aplicamos cada función a cada matriz
log(x)
log2(x)
log10(x)
log1p(x)
cos(x)
cosh(x)
acos(x)
acosh(xS)
sin(x)
sinh(x)
asin(x)
asinh(xS)
tan(x)
atan(x)
tanh(x)
atanh(xS)
sqrt(y)
abs(y)
sign(y)
ceiling(y)
floor(y)
cumsum(x)
cumprod(x)
exp(x)
expm1(x)

log(xS)
log2(xS)
log10(xS)
log1p(xS)
cos(xS)
cosh(xS)
acos(xS)
acosh(xS)
sin(xS)
sinh(xS)
asin(xS)
asinh(xS)
tan(xS)
atan(xS)
tanh(xS)
atanh(xS)
sqrt(yS)
abs(yS)
sign(yS)
ceiling(yS)
floor(yS)
cumsum(xS)
cumprod(xS)
exp(xS)
expm1(xS)

log(y)
log2(y)
log10(y)
log1p(y)
cos(y)
cosh(y)
acos(y)
acosh(yS)
sin(y)
sinh(y)
asin(y)
asinh(yS)
tan(y)
atan(y)
tanh(y)
atanh(yS)
sqrt(y)
abs(y)
sign(y)
ceiling(y)
floor(y)
cumsum(y)
cumprod(y)
exp(y)
expm1(y)

log(yS)
log2(yS)
log10(yS)
log1p(yS)
cos(yS)
cosh(yS)
acos(yS)
acosh(yS)
sin(yS)
sinh(yS)
asin(yS)
asinh(yS)
tan(yS)
atan(yS)
tanh(yS)
atanh(yS)
sqrt(yS)
abs(yS)
sign(yS)
ceiling(yS)
floor(yS)
cumsum(yS)
cumprod(yS)
exp(yS)
expm1(yS)

library(GPUmatrix)

# Crear dos matrices en la GPU
A <- gpu.matrix(1:4, nrow = 2, dtype = "float64")
B <- gpu.matrix(c(2, 4, 6, 8), nrow = 2)

# Realizar algunas operaciones algebraicas
C <- A + B
D <- A %*% B

# Verificar que los resultados son correctos
print(C)
#>      [,1] [,2]
#> [1,]    3    7
#> [2,]    5    9

print(D)
#>      [,1] [,2]
#> [1,]   14   20
#> [2,]   30   44

# Aplicar una función a una matriz en la GPU
E <- gpu.matrix(c(1, 2, 3, 4), nrow = 2)
F <- log(E)

# Verificar que el resultado es correcto
print(F)
#>           [,1]      [,2]
#> [1,] 0.0000000 0.6931472
#> [2,] 1.0986123 1.3862944



