import pstats
stats = pstats.Stats("explorer_profile.out")
stats.sort_stats("cumulative").print_stats(20)
