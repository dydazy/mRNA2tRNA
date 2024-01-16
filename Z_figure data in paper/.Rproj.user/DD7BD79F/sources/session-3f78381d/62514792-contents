library(AnnotationDbi)
library(org.Hs.eg.db)
library(clusterProfiler)
library(dplyr)
library(ggplot2)
library(R.utils)
  
R.utils::setOption('clusterProfiler.download.method','auto') 

diff=read.csv("DATA/fig3/top10%_feature_intersection_gene_list.csv")
gene.df <- bitr(diff$intersection ,fromType="ENSEMBL",toType="ENTREZID", OrgDb = org.Hs.eg.db)                    
gene <- gene.df$ENSEMBL
  
library(enrichplot)
ego_MF <- enrichGO(gene = gene,
                     OrgDb=org.Hs.eg.db,
                     keyType = "ENSEMBL",
                     ont = "BP",
                     pAdjustMethod = "BH",
                     pvalueCutoff = 0.001,
                     qvalueCutoff = 0.001,
                     readable = TRUE)
  
egosimp <- simplify(ego_MF,cutoff=1,by="p.adjust",select_fun=min,measure="Wang")
  
emapplot(pairwise_termsim(egosimp), showCategory = 25, color = "p.adjust", layout.params = list(layout = "nicely"),
           edge.params=list(category_node = 10,
                            category_label = 10,
                                line = 4,min = 0.001),
           node_scale = 0.1,
           line_scale = 0.1,
           node_label_size = 0.1)

diff=read.csv("DATA/fig3/top5%_feature_intersection_gene_list.csv")
gene.df <- bitr(diff$intersection ,fromType="ENSEMBL",toType="ENTREZID", OrgDb = org.Hs.eg.db)                     
gene <- gene.df$ENTREZID
library(enrichplot)
ego_MF <- enrichGO(gene = gene,
                   OrgDb=org.Hs.eg.db,
                   keyType = "ENTREZID",
                   ont = "BP",
                   pAdjustMethod = "BH",
                   pvalueCutoff = 0.001,
                   qvalueCutoff = 0.001,
                   readable = TRUE)

egosimp <- simplify(ego_MF,cutoff=1,by="p.adjust",select_fun=min,measure="Wang")

emapplot(pairwise_termsim(egosimp), showCategory = 25, color = "p.adjust", layout.params = list(layout = "nicely"),
         edge.params=list(category_node = 10,
                          category_label = 10,
                          line = 4,min = 0.001),
         node_scale = 0.001,
         line_scale = 0.001,
         node_label_size = 0.1)
