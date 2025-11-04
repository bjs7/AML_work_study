params = {
    "num_threads": 4,
    #"time_window": 21600,
    #"time_window": 86400,

    "vertex_stats": True,         # produce vertex statistics
    "vertex_stats_cols": [3,4],     # produce vertex statistics using the selected input columns
    
    # features: 0:fan,1:deg,2:ratio,3:avg,4:sum,5:min,6:max,7:median,8:var,9:skew,10:kurtosis
    "vertex_stats_feats": [0, 1, 2, 3, 4, 8, 9, 10],  # fan,deg,ratio,avg,sum,var,skew,kurtosis

    # scatter gather parameters
    "scatter-gather": True,
    "scatter-gather_tw": 21600,
    "scatter-gather_bins": [y+2 for y in range(2)],

    # length-constrained simple cycle parameters
    "lc-cycle": False,
    "lc-cycle_tw": 86400,
    "lc-cycle_len": 10,
    "lc-cycle_bins": [y+2 for y in range(2)],

    # fan in/out parameters
    "fan": True,
    "fan_tw": 86400,
    "fan_bins": [y+2 for y in range(2)],
    
    # in/out degree parameters
    "degree": True,
    "degree_tw": 86400,
    "degree_bins": [y+2 for y in range(2)],

    # temporal cycle parameters
    "temp-cycle": True,
    "temp-cycle_tw": 86400,
    "temp-cycle_bins": [y+2 for y in range(2)],
}

