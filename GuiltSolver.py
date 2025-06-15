import os
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import linecache
import neat
import random


inc = 0

maxEpochs = 200
RPM = 10
#rpm up
#population up?

avg_fitness_history = []
avg_coop_history    = []
avg_defect_history  = []
split_rate_history  = []
steal_rate_history  = []

#this is the payoff dictionary, you look up the keys on the left with the outputs of the net and the dictionary returns payoffs (on the right)
payoff = {
    (0, 0): (3.5, 3.5),   # both cooperate
    (0, 1): (0, 5),   # you coop, opp defect
    (1, 0): (5, 0),   # you defect, opp coop
    (1, 1): (1, 1),   # both defect
}

    
def playMatch(genome_a, genome_b, config,roundsPerMatch = RPM):
    net_a = neat.nn.FeedForwardNetwork.create(genome_a, config)
    net_b = neat.nn.FeedForwardNetwork.create(genome_b, config)

    history_a = []
    history_b = []
    score_a = 0
    score_b = 0
    coop_actions = 0
    defect_actions = 0
    both_defects = 0
    splits = 0
    steals = 0
    
    for events in range(roundsPerMatch):
        
        #This is what to change for nuance, passing whats below gives both the previous action of both agents as input for each agent
        #inp_a = [ history_a[-1] if history_a else random.randint(0, 1), history_b[-1] if history_b else random.randint(0, 1) ]
        #inp_b = [ history_b[-1] if history_b else random.randint(0, 1),history_a[-1] if history_a[-1] if history_a else random.randint(0, 1) ]

        
        inp_a = [ history_a[-1] if history_a else 0, history_b[-1] if history_b else 0 ]
        inp_b = [ history_b[-1] if history_b else 0, history_a[-1] if history_a else 0 ]

        #gets output from the network
        out_a = net_a.activate(inp_a)
        out_b = net_b.activate(inp_b)
        

        #if the net is closer to 0, output 0 and cooperate
        #if the net is closer to 1, output 1 and defect
        act_a = 0 if (out_a[0] + inc) > out_a[1] else 1
        act_b = 0 if (out_b[0] + inc) > out_b[1] else 1

        
        pa, pb = payoff[(act_a, act_b)]
        score_a += pa
        score_b += pb


        # count splits and steals
        if act_a == 0 and act_b == 0:
            splits += 1
        elif act_a == 1 and act_b == 1:
            both_defects += 1
        else:
            steals += 1

        coop_actions   += (1 - act_a) + (1 - act_b)
        defect_actions += act_a + act_b


        #adds action to history
        history_a.append(act_a)
        history_b.append(act_b)

        
    return(score_a, score_b, coop_actions, defect_actions,splits,steals, both_defects)
        
        
def eval_genomes(genomes, config):
    random.shuffle(genomes)
    
    total_splits           = 0          
    total_steals           = 0        
    total_both_defects     = 0         
    total_coop_actions     = 0        
    total_defect_actions   = 0      
    total_coop_rate        = 0.0
    total_defect_rate      = 0.0
    matches_count          = 0
    
    for _, genome in genomes:
        genome.fitness = 0


    
    # round‑robin (i<j) so each pair plays once
    for i in range(len(genomes)):
        id_i, genome_i = genomes[i]
        
        for j in range(i+1, len(genomes)):
            id_j, genome_j = genomes[j]
            score_i, score_j, coop_c, defect_c, splits_c, steals_c, both_c = playMatch(genome_i, genome_j, config)
            genome_i.fitness += score_i
            genome_j.fitness += score_j

            total_coop_actions   += coop_c
            total_defect_actions += defect_c
            total_splits         += splits_c
            total_steals         += steals_c
            total_both_defects   += both_c
            matches_count        += 1

    
    #normalize by number of opponents
    num_opponents = len(genomes) - 1
    
    for _, genome in genomes:
        genome.fitness /= num_opponents
        
    avg_fit = np.mean([g.fitness for _, g in genomes])
    avg_fitness_history.append(avg_fit)

    denom_actions = matches_count * 2 * RPM     # total individual moves this gen
    denom_rounds  = matches_count * RPM         # total rounds this gen

    avg_coop   = total_coop_actions   / denom_actions
    avg_defect = total_defect_actions / denom_actions
    avg_coop_history.append(avg_coop)
    avg_defect_history.append(avg_defect)

    split_rate_history.append(total_splits / denom_rounds)
    steal_rate_history.append(total_steals / denom_rounds)
    

#if you have to stop the program manually you can go from a checkpoint by uncommenting the first line
#in the function below (and swapping the placeholder checkpoint) and commenting in the second 
def run_neat(config):
#p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-85')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, maxEpochs)

    means     = stats.get_fitness_mean()      # mean fitness per gen
    sizes     = stats.get_species_sizes()     # [[size per species] … ]
    fitnesses = stats.get_species_fitness()   # [[fitness per species] … ]
        
    import matplotlib.pyplot as plt

    # ─── 1) Grab the raw per-generation species stats ────────────────
    # stats is your StatisticsReporter; its generation_statistics is
    # a list (one entry per gen) of dicts: { species_id: {genome_id: fitness, ...}, ... }
    gen_stats = stats.generation_statistics

    # ─── 2) Compute the three new series purely in Python ─────────────
    # how many species each generation?
    species_counts = [ len(g) for g in gen_stats ]

    # average species size (number of genomes in each species, averaged over species)
    avg_species_size = []
    for g in gen_stats:
        sizes = [ len(members) for members in g.values() ]
        avg_species_size.append(sum(sizes) / len(sizes) if sizes else 0)

    # average species fitness (per-species average fitness, then averaged over species)
    avg_species_fitness = []
    for g in gen_stats:
        # for each species, compute its mean fitness
        mean_fits = []
        for members in g.values():
            vals = list(members.values())
            mean_fits.append(sum(vals) / len(vals) if vals else 0)
        avg_species_fitness.append(sum(mean_fits) / len(mean_fits) if mean_fits else 0)

    # ─── 3) Plot all five panels ──────────────────────────────────────
    gens = range(1, len(avg_fitness_history) + 1)

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 18), sharex=True)

    # Panel 1–4: exactly as you already have them
    ax1.plot(gens, avg_fitness_history,   label='Avg Fitness', linewidth=2)
    ax1.set_ylabel('Fitness');       ax1.legend(loc='upper left')

    ax2.plot(gens, avg_coop_history,   '--', label='Coop Rate')
    ax2.plot(gens, avg_defect_history, ':',  label='Defect Rate')
    ax2.set_ylabel('Rate');           ax2.legend(loc='upper left')

    ax3.plot(gens, split_rate_history, '--', label='Split Rate')
    ax3.plot(gens, steal_rate_history, ':',  label='Steal Rate')
    ax3.set_ylabel('Rate');           ax3.legend(loc='upper left')

    ax4.plot(gens, species_counts,   label='Species Count')
    ax4.set_ylabel('Num Species');   ax4.legend(loc='upper left')

    # Panel 5: Avg Species Size & Fitness

    # ─── Now overlay the top‐lifespan species on Panel 5 ─────────
    TOP_N = 10
    gen_stats = stats.generation_statistics

    # find the top N by lifespan
    lifes = {}
    for gen_idx, species_map in enumerate(gen_stats, start=1):
        for sid in species_map:
            lifes.setdefault(sid, []).append(gen_idx)
    sp_lifespans = {sid: max(gens) - min(gens) + 1 for sid, gens in lifes.items()}
    top_species  = sorted(sp_lifespans, key=lambda s: sp_lifespans[s], reverse=True)[:TOP_N]

    # build size series & plot
    for sid in top_species:
        series = []
        for gen_idx, species_map in enumerate(gen_stats, start=1):
            series.append(len(species_map.get(sid, {})))
        ax5.plot(gens, series, linestyle='-', linewidth=2,
                 label=f"Sp {sid} (life={sp_lifespans[sid]})")

    ax5.set_xlabel('Generation')
    ax5.set_ylabel('Species Metrics')


    plt.tight_layout()
    plt.show()
####THIS IS ALL CODE FOR THE GRAPH
###-----------------------------------------------------------------------------------------------------
##
##
##
##    sns.set_style("whitegrid")
##
##
##    fitness_scores = stats.get_fitness_mean()
##    generations = np.arange(1, len(fitness_scores) + 1)
##
##
##    plt.figure(figsize=(10, 6))
##
##
##    sns.lineplot(x=generations, y=fitness_scores, marker='o', linewidth=2.5, markersize=10, color='teal')
##
##
##    plt.title('Average Final Balance of Populations Over Generations', fontsize=18, fontweight='bold', color='teal')
##    plt.xlabel('Generation', fontsize=14, fontweight='bold', color='teal')
##    plt.ylabel('Average Final Balance of Population', fontsize=14, fontweight='bold', color='teal')
##
##
##    ticks = np.arange(1, max(generations)+1, 50)
##    plt.xticks(ticks, fontsize=12, fontweight='bold', color='teal')
##    plt.yticks(fontsize=12, fontweight='bold', color='teal')
##
##
##    max_fitness = max(fitness_scores)
##    max_gen = generations[np.argmax(fitness_scores)]
##    plt.annotate(f'Peak ({max_gen}, {max_fitness:.2f})', xy=(max_gen, max_fitness), xytext=(max_gen+0.5, max_fitness+0.5),
##                 arrowprops=dict(facecolor='teal', shrink=0.05), fontsize=12, color='teal')
##
##    plt.tight_layout()
##    plt.show()

#the bit that runs it

if __name__ == '__main__':
    
    local_dir   = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    run_neat(config)
