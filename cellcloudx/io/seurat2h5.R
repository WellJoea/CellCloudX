library(Seurat)
library(SeuratData)

save_h5 <-function(seuobj, filename ='sc',use_dior=FALSE,
                   use_rna_data=FALSE,
                   save_counts = TRUE,
                   assay.name='Spatial'){
    # seuobj@meta.data$nCount_RNA <- colSums(x = seuobj, slot = "counts") 
    # seuobj@meta.data$nFeature_RNA <- colSums(x = GetAssayData(object = seuobj, slot = "counts") > 0) 

    if(use_dior){
        dior::write_h5(seuobj, 
                       file= paste0(filename, '.h5'), 
                       object.type = 'seurat',
                       assay.name =assay.name)  
    }else{
        DefaultAssay(seuobj) <- assay.name
        tryCatch({seuobj@reductions$pca@global = TRUE}, error= function(e){''})
        if (use_rna_data){
            seuobj@assays[[assay.name]]@data = seuobj@assays$RNA@data 
        }
        if (save_counts){
             seuobj@assays[[assay.name]]@data =  seuobj@assays[[assay.name]]@counts
        }
        SeuratDisk::SaveH5Seurat(seuobj, 
                                 check.names = FALSE,
                                 filename =  paste0(filename, '.h5Seurat'),
                                 overwrite=TRUE)
        SeuratDisk::Convert(paste0(filename, '.h5Seurat'),
                            dest = "h5ad", 
                            overwrite=TRUE,
                            check.names = FALSE,
                            assay = assay.name)
        write.csv(seuobj@meta.data,
                  paste0(filename, '.meta.csv'))
    }
}

read_h5ad2rds <- function( h5adfile, tmpdir = NULL, remove_tmp=TRUE, state_counts=FALSE){
    if (is.null(tmpdir)){
        datadir <- dirname(h5adfile)
    }else{
        datadir <- tmpdir
    }
    setwd(datadir)

    SeuratDisk::Convert(h5adfile, dest = "h5seurat", overwrite = TRUE)
    seuobj <- SeuratDisk::LoadH5Seurat(gsub("h5ad","h5seurat", h5adfile), meta.data=FALSE) # ERROR in levels factor

    if (remove_tmp & (file.exists(gsub("h5ad","h5seurat", h5adfile)))){
      file.remove(gsub("h5ad","h5seurat", h5adfile))
    }

    scanpyobj <- anndata::read_h5ad(h5adfile)
    metadata  <- scanpyobj$obs
    seuobj@meta.data <- metadata[rownames(seuobj@meta.data),]

    if (state_counts){
        seuobj@meta.data$nCount_RNA <- colSums(x = seuobj, slot = "counts") 
        seuobj@meta.data$nFeature_RNA <- colSums(x = GetAssayData(object = seuobj, slot = "counts") > 0) 
    
        seuobj <- PercentageFeatureSet(seuobj, pattern = "^MT-|Mt-|mt-",      col.name = "percent.mt")
        seuobj <- PercentageFeatureSet(seuobj, pattern = "^RP[SL]|^Rp[sl]",   col.name = "percent.ribo")
        seuobj <- PercentageFeatureSet(seuobj, pattern = "^HB[^(P)]|^Hb[^(p)]", col.name = "percent.hb")
    }
    seuobj
}


imshow <- function(img, use_raster=TRUE){
    if (use_raster){
        plot(raster::as.raster(img))
    }else{
        recolorize::plotImageArray(img, main = "RGB image") 
    }
}

save_np <- function(data, file){
    library(reticulate)
    np = import("numpy")
    np$save(file,r_to_py(data))
} 

image_df <-function(seust, filename=NULL){
    slice <- Seurat::Images(seust)
    images <- seust[[slice]]
    sf_info <- c(images@scale.factors,
                  spot.radius= images@spot.radius,
                  assay = images@assay, 
                  key = images@key) 
    
    if (!is.null(filename)){
        jsonlite::write_json(sf_info, 
                             paste0(filename, '.scale.factors.json'),
                             digits=NA,
                             pretty = TRUE,
                             auto_unbox = TRUE
                            ) 
        write.csv(images@coordinates, paste0(filename, '.coordinates.csv'))
        save_np(images@image, paste0(filename, '.image.npy'))
    }
    sf_info
}

seu5toh5 <- function(seurobj, filename, assay = 'Xenium'){
    library(hdf5r)
    file.h5 <- H5File$new(filename, mode="w")

    DefaultAssay(seurobj) <- assay
    Data = Seurat::GetAssayData(seurobj)
    feature = rownames(Data)
    cells = colnames(Data)
    spatial = data.frame(seurobj@images$fov@boundaries$centroids@coords)
    rownames(spatial) <- seurobj@images$fov@boundaries$centroids@cells
    spatial = spatial[cells, ]
    metadata = seurobj@meta.data
    metadata = data.frame(sapply(metadata[cells, ], as.character))

    file.h5$create_group(assay)
    file.h5[[paste0(assay, "/counts")]] <- as.matrix(Data)
    file.h5[[paste0(assay, "/feature")]] <- feature
    file.h5[[paste0(assay, "/cells")]] <- cells
    file.h5[[paste0(assay, "/spatial")]] <- spatial
    file.h5[[paste0(assay, "/meta")]] <- metadata

    file.h5$close_all()
}