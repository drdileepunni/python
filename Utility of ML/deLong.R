getwd()
preds = read.csv('/home/dileep/Desktop/python/Nature/predictions.csv')

install.packages("pROC")
require('pROC')

roc_apache <- roc(preds$outcome, preds$apache)
roc_trop <- roc(preds$outcome, preds$tropics)
roc_sgb <- roc(preds$outcome, preds$sgb)
roc_gb <- roc(preds$outcome, preds$gb)
roc_rf <- roc(preds$outcome, preds$rf)
roc_lr <- roc(preds$outcome, preds$lr)
roc_xgb <- roc(preds$outcome, preds$xgb)
roc_nn <- roc(preds$outcome, preds$nn)

mod_roc_ls = list(roc_sgb, roc_gb, roc_rf, roc_lr, roc_xgb, roc_nn)

pvals_ap = list()
pvals_trop = list()
c = 1
for (roc in mod_roc_ls) {
  apache_dl <- roc.test(roc_apache, roc, method = 'delong')
  tropics_dl <- roc.test(roc_trop, roc, method = 'delong')
  pvals_ap[c] <- apache_dl["p.value"]
  pvals_trop[c] <- tropics_dl["p.value"]
  c = c+1
}

pvals_ap
pvals_trop

pvals_ap = unlist(pvals_ap, recursive = F)
pvals_trop = unlist(pvals_trop, recursive = F)

df = data.frame(pvals_ap, pvals_trop)
write.csv(df, 'deLong.csv')
