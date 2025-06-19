import random
from typing import List, Tuple, Dict
import numba as jit
import numpy as np
from numba import njit

class Player:
    __slots__ = ['id', 'role', 'suspicion_scores', 'genes', 'ratings', 'probabilities']
    
    def __init__(self, id, role, genes):
        self.id = id
        self.role = role
        self.genes = genes.astype(np.float32)  # Ensure float32
        

def update_suspicion(resistance: Player,
                    players: List[Player], 
                    team: List[Player], 
                    mission_leader_idx: int,
                    team_size: int, 
                    sabotage_count: int, 
                    mission_num: int, 
                    player_votes: Dict[int, int], 
                    mission_successful: bool, 
                    voting_round: int,
                    verbose: bool = False) -> None:
  
    team_ids = {p.id for p in team}
    resistance_in_team = resistance.id in team_ids
    
    for suspect in (p for p in players if p != resistance):
        suspect_id = suspect.id
        suspect_in_team = suspect_id in team_ids
        is_leader = (mission_leader_idx == suspect_id)
        
        ratings = []
        
        # ====== Spy Detection (SD) Cases ======
        if suspect_in_team:
            if (team_size == 2 and (
                (resistance_in_team and sabotage_count == 1) or 
                (not resistance_in_team and sabotage_count == 2))):
                resistance.suspicion_scores[suspect_id] += 1000  # Absolute certainty
                if verbose: print(f"######Player {resistance.id}, identified player {suspect_id} as a spy! ######")
                continue
            
            if (team_size == 3 and resistance_in_team 
                and sabotage_count == 2):
                resistance.suspicion_scores[suspect_id] += 1000
                if verbose: print(f"######Player {resistance.id}, identified player {suspect_id} as a spy! ######")
                continue

            # ====== Team Member Ratings (r1-r7) ======
            if team_size == 2:
              if not resistance_in_team and sabotage_count == 1:
                ratings.append(0)  # r1
              elif resistance_in_team and sabotage_count == 0:
                ratings.append(3)  # r4
              elif not resistance_in_team and sabotage_count == 0:
                ratings.append(4)  # r5
                
            elif team_size == 3:
              if resistance_in_team and sabotage_count == 1:
                ratings.append(0)  # r1
              elif not resistance_in_team and sabotage_count == 1:
                ratings.append(1)  # r2
              elif not resistance_in_team and sabotage_count == 2:
                ratings.append(2)  # r3
              elif resistance_in_team and sabotage_count == 0:
                ratings.append(5)  # r6
              elif not resistance_in_team and sabotage_count == 0:
                ratings.append(6)  # r7
                if is_leader:
                    ratings.append(7)  # r8
        
        
        # ====== Leader Ratings (r8-r9) ======
        if is_leader:
            if sabotage_count == 0:
                if (team_size == 3 and resistance_in_team) or team_size == 2:
                    ratings.append(8)  # r9
            if sabotage_count >= 1:
                ratings.append(9)  # r10
        
        # ====== Voting Behavior (r11-r14) ======
        vote = player_votes[suspect_id]
        if vote == 0:  # 'No' vote
            if voting_round == 5:
                ratings.append(10)  # r11
            if mission_successful:
                ratings.append(11)  # r12
            else:
                ratings.append(12)  # r13
        if team_size == 3 and not suspect_in_team and vote == 1:
          ratings.append(13)  # r14
        
        # Apply all ratings
        for rating_idx in ratings:
            resistance.suspicion_scores[suspect_id] += resistance.genes[rating_idx]

def sabotage(spy: Player, team: List[Player], mission_num: int,
             mission_completed: int, mission_sabotaged: int, random_behavior: bool = False) -> bool:
    """
    Determines if a specific spy chooses to sabotage.
    
    Args:
        spy: The spy making the decision
        team: Current mission team (including this spy)
        mission_num: Mission number (1-5)
        mission_completed: Number of successful missions
        mission_sabotaged: Number of failed missions
        
    Returns:
        True if this spy decides to sabotage, False otherwise
    """
    # If random behavior is True, sabotage with 50% probability
    if random_behavior:
      return np.random.random() < 0.5

    assert len(spy.genes) == 10, "Spy genes should have length 10"

    # Win/loss conditions override everything
    if mission_sabotaged == 2 or mission_completed == 2:
      #print('triggered: mission_sabotage == 2 or mission_completed == 2')
      return True
    
    # Count spies in team (excluding self if needed)
    spy_count = sum(1 for p in team if p.role == 1)
    
    # Extract this spy's thresholds
    s7, s8, s9, s10 = spy.genes[6:10]
    p = np.random.random()
    
    # Apply Algorithm 5's rules
    if spy_count == 2 and len(team) == 2:  # 2-player team with 2 spies
      #if p < s7: print('triggered: s7')
      return p < s7
    elif spy_count == 2 and len(team) == 3:  # 3-player team with 2 spies
      #if p < s8: print('triggered: s8')
      return p < s8
    elif mission_num == 1:  # First mission special case
      #if p < s9: print('triggered: s9')
      return p < s9
    else:  # Default case
      #if p < s10: print('triggered: s10')
      return p < s10
    
def spy_vote(spy: Player, mission_leader_idx: int, team: List[Player], mission_num: int, 
             completed_missions: int, sabotaged_missions: int,
             voting_round: int, random_behavior: bool = False) -> bool:
    """
    Spy voting behavior according to Algorithm 4.
    
    Args:
        spy: The spy player voting
        mission_leader_idx: Index of the mission leader
        team: Proposed team members
        mission_num: Current mission number (1-5)
        completed_missions: Number of successfully completed missions
        sabotaged_missions: Number of sabotaged missions
        voting_round: Current voting round (1-5)
        
    Returns:
       int: 1 for 'Yes', 0 for 'No'
    """
    # If random behavior is True, vote with 50% probability 
    if random_behavior:
      return np.random.random() < 0.5

    # Count spies in team (including self if leader)
    spies_in_team = sum(1 for p in team if p.role == 1)
    
    # Algorithm conditions (ordered by precedence)
    if spy.id == mission_leader_idx or mission_num == 1:
        return True
    elif completed_missions == 2:
        return spies_in_team > 0
    elif sabotaged_missions == 2:
        return spies_in_team > 0
    elif voting_round == 5:
        return np.random.random() < spy.genes[2]
    elif spies_in_team == 1:
        return np.random.random() < spy.genes[3] 
    elif spies_in_team == 2:
        return np.random.random() < spy.genes[4] 
    else:  # no spies in team
        return np.random.random() < spy.genes[5] 
    
def resistance_vote(voter: Player, team: List[Player], mission_leader_idx: int, mission_num: int, 
                  voting_round: int, all_players: List[Player], random_behavior: bool = False) -> bool:
    """
    Correct Resistance voting using proper suspicion_scores lookup.
    
    Args:
        voter: The Player casting the vote (uses THEIR suspicion_scores)
        team: Proposed team members
        mission_num: Current mission number (1-5)
        voting_round: Current voting round (1-5)
        all_players: Complete list of players in the game
        
    Returns:
        bool: True for 'Yes', False for 'No'
    """
    # If random behavior is True, vote with 50% probability
    if random_behavior:
      return np.random.random() < 0.5

    # Automatic yes conditions (early exit)
    if (voter.id == mission_leader_idx or               # Is team member
        voting_round == 5 or                            # Last chance
        mission_num == 1):                              # First mission
        return 1
    
    # Get top 2 most suspicious players FROM VOTER'S PERSPECTIVE
    top_two = sorted(
        all_players,
        key=lambda p: voter.suspicion_scores[p.id],  # Lookup by player ID
        reverse=True
    )[:2]
    
    # Vote Yes ONLY if NEITHER top suspicious is in team
    return 1 if not any(p in team for p in top_two) else 0

def propose_team(leader: Player, players: List[Player], team_size: int, mission_num: int, random_behavior: bool = False) -> List[Player]:
    """
    Final optimized team selection matching EXACT Player class structure.
    Uses suspicion_scores array and maintains proper player references.
    """
    # If random behavior is True, choose random team with uniform probability
    if random_behavior:
      return np.random.choice(players, size=team_size, replace=False)

    team = [leader]  # Leader always included
    other_players = [p for p in players if p != leader]
    player_ids = [p.id for p in players]  # Get all IDs
    
    if leader.role == 0:  # RESISTANCE
        if mission_num == 1:
            # Mission 1: Uniform random selection
            team.extend(np.random.choice(other_players, size=team_size-1, replace=False))
        else:
            # Sort players by their suspicion score FROM THE LEADER'S PERSPECTIVE
            # leader.suspicion_scores[i] = leader's suspicion of player i
            sorted_ids = sorted(
                [p.id for p in other_players],
                key=lambda pid: leader.suspicion_scores[pid]
            )[:team_size-1]
            # Convert back to player objects
            team.extend([players[pid] for pid in sorted_ids])
    
    else:  # SPY
        p = np.random.random()
        resistance = [p for p in other_players if p.role == 0]
        
        if team_size == 2:
            if p < leader.genes[0]:  # s1
                team.append(np.random.choice(other_players))
            elif resistance:
                team.append(np.random.choice(resistance))
            else:  # Only spies left
                team.append(np.random.choice([p for p in other_players if p.role == 1]))
        else:  # team_size == 3
            if p < leader.genes[1]:  # s2
                team.extend(np.random.choice(other_players, size=2, replace=False))
            else:
                # Add resistance first
                add = min(2, len(resistance))
                team.extend(np.random.choice(resistance, size=add, replace=False))
    return team

def setup_game(genes_list: List[np.ndarray], random_behavior: bool = False) -> List[Player]:
    """
    Optimized game setup without redundant conversions.
    
    Args:
        genes_list: List of numpy arrays (2 spies with size 10, 3 resistance with size 14)
        
    Returns:
        List of Player objects with assigned IDs, roles, and shuffled genes
    """
    if random_behavior:
       roles = [0, 0, 0, 1, 1]
       random.shuffle(roles)
       players = [Player(i, role=role, genes=np.zeros(14)) for i, role in  enumerate(roles)]
       return players

    # Get lengths in one vectorized operation
    lengths = np.array([genes.shape[0] for genes in genes_list])  # shape[0] is faster than len()
    
    # Vectorized role assignment
    roles = (lengths == 10).astype(np.int8)  # 1 for spy, 0 for resistance
    
    # Pre-allocate players list
    n_players = len(genes_list)
    players = [None] * n_players
    
    # Shuffle indices - more efficient than shuffling genes_list
    indices = np.random.permutation(n_players)
    
    for new_pos, original_idx in enumerate(indices):
        players[new_pos] = Player(
            id=new_pos,
            role=roles[original_idx],
            genes=genes_list[original_idx]  # Use original list directly
        )
    
    return players

def simulate_game(list_of_players: List[Player], verbose=False, random_behavior: bool = False) -> str:
    mission_sizes = [2, 3, 2, 3, 3]
    players = setup_game(list_of_players, random_behavior=random_behavior)
    complete_missions = 0
    sabotaged_missions = 0
    mission_leader_idx = random.randint(0, 4) if random_behavior else 0  # Track leader by index instead of object
    winner = None
    
    spies = [p for p in players if p.role == 1]
    resistance = [p for p in players if p.role == 0]

    
    if verbose: 
        print("\n======= NEW GAME STARTING =======")
        print(f"Roles: {[str(p.id) + ': ' + ('Resistance' if p.role == 0 else 'Spy') for p in players]}")




    for mission_num, team_size in enumerate(mission_sizes, 1):
        if verbose: print(f"\n####### MISSION {mission_num} (Size {team_size}) #######")
        
        # Voting Phase
        voting_round = 1
        mission_successful = False

        while voting_round <= 5:
            if verbose: print(f"\n--- Voting Round {voting_round} ---")
            
            # Team Selection
            team = propose_team(players[mission_leader_idx], players, team_size, mission_num, random_behavior)
            if verbose: 
              print(f"Mission Leader: Player {players[mission_leader_idx].id}")
              print(f"Proposed Team: {[str(p.id) + ': ' + ('Resistance' if p.role == 0 else 'Spy') for p in team]}")
            
            if random_behavior:
               break
            # Voting
            vote_results = {}  # {player_id: int}
            for p in players:
              vote_results[p.id] = spy_vote(p, 
                                            mission_leader_idx,
                                            team, 
                                            mission_num, 
                                            complete_missions, 
                                            sabotaged_missions, 
                                            voting_round,
                                            random_behavior
                                            ) if p in spies else resistance_vote(
                                                 p,
                                                 team,
                                                 mission_leader_idx,
                                                 mission_num, 
                                                 voting_round, 
                                                 players,
                                                 random_behavior
                                                 )
            
            if verbose:
              print(f"Votes: {[str(pid) + ': ' + ('Yes' if vote == 1 else 'No') for pid, vote in vote_results.items()]}")
            
            if sum(vote_results.values()) >= 3:
              if verbose: print("Team approved!")
              break
            else:
              # Rotate leader if team rejected
              voting_round += 1
              mission_leader_idx = (mission_leader_idx + 1) % 5
              if verbose: print(f"Rotate leader!")
            

        if voting_round > 5:
            if verbose: print("Game over: team rejected 5 times")
            return 'spies'

        # Mission Execution
        sabotage_count = sum(1 for p in team if p in spies and sabotage(p, team, mission_num, complete_missions, sabotaged_missions, random_behavior))
        if sabotage_count >= 1:
            if verbose: print(f"Number of sabotages: {sabotage_count}")
            sabotaged_missions += 1
            if verbose: print("Mission Failed! (Sabotaged)")
        else:
            complete_missions += 1
            mission_successful = True
            if verbose: print("Mission Succeeded!")

        if verbose: print(f"Scores so far: sabotaged missions = {sabotaged_missions} / completed missions: {complete_missions}")


        # Win Check
        if complete_missions >= 3:
            if verbose: print("Resistance wins - 3 successful missions!")
            return 'resistance'
        elif sabotaged_missions >= 3:
            if verbose: print("Spies win - 3 sabotaged missions!")
            return 'spies'

        if not random_behavior:
            for p in resistance:
                update_suspicion(p,
                                 players,
                                 team,
                                 mission_leader_idx, 
                                 team_size, 
                                 sabotage_count, 
                                 mission_num, 
                                 vote_results,  
                                 mission_successful,
                                 voting_round)

        # Next mission   
        mission_num += 1   
        # Rotate leader for next mission
        mission_leader_idx = (mission_leader_idx + 1) % 5

    # If all missions complete without winner (shouldn't happen with 5 missions)
    return winner

#################################### TESTING #####################################
resistance_pop = np.random.uniform(-1, 1, (100, 14)).astype(np.float32)  # 3x14 array
spy_pop = np.random.uniform(0, 1, (100, 10)).astype(np.float32)  # 2x10 array

r = random.choices(resistance_pop, k=3)
s = random.choices(spy_pop, k=2)

players = r + s
win = 0

for i in range(100_000):
  winner = simulate_game(players, verbose=False, random_behavior=True)
  if winner == 'resistance':
    win += 1

print(f"Resistance win rate random game: {win/100_000:.3f}")