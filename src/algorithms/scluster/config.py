"""Config for the Spectral Clustering algorithm."""

# Spectral Clustering Params
config = {
    "num_layers" : 20, # 20 (n°1)
    "scluster_k" : 9,  
    #"evec_smooth": 16, # 32 (n°1)
    #"rec_smooth" : 16, # 16 (n°1)
    #"rec_width"  : 1,

    #"evec_smooth": 4,
    #"rec_smooth" : 16,
    #"rec_width"  : 2,
    # 

    "evec_smooth": 3,
    "rec_smooth" : 9,
    "rec_width"  : 2, 
     
   # "evec_smooth": 5,
   # "rec_smooth" : 9,
   # "rec_width"  : 2,
   #  
    #"evec_smooth": 3,
    #"rec_smooth" : 16,
    #"rec_width"  : 2,

    "hier" : True
}

algo_id = "scluster"
is_boundary_type = True
is_label_type = True
