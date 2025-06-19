import random
from typing import List, Tuple, Dict
import numpy as np

class Player:
    __slots__ = ['id', 'role', 'suspicion_scores', 'genes']
    
    def __init__(self, id, role, genes):
        self.id = id
        self.role = role
        self.genes = genes  
        
        if role == 0:  # Resistance
            self.suspicion_scores = np.zeros(5, dtype=np.float32)
            self.suspicion_scores[id] = -1000.0

def update_suspicion(resistance: Player,
                    players: List[Player], 
                    team: List[Player], 
                    mission_leader_idx: int,
                    team_size: int, 
                    sabotage_count: int, 
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
             mission_completed: int, mission_sabotaged: int) -> bool:
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
             voting_round: int) -> bool:
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
       bool: True for 'Yes', False for 'No'
    """
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
                  voting_round: int, all_players: List[Player]) -> bool:
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

def propose_team(leader: Player, players: List[Player], team_size: int, mission_num: int) -> List[Player]:
    """
    Final optimized team selection matching EXACT Player class structure.
    Uses suspicion_scores array and maintains proper player references.
    """
    team = [leader]  # Leader always included
    other_players = [p for p in players if p != leader]
    
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
            else:
                team.append(np.random.choice(resistance))
        else:  # team_size == 3
            if p < leader.genes[1]:  # s2
                team.extend(np.random.choice(other_players, size=2, replace=False))
            else:
                team.extend(np.random.choice(resistance, size=2, replace=False))
    return team

import random
import numpy as np
from typing import List

def setup_game(genes_list: List[np.ndarray]) -> List[Player]:
    """Initialize the game by creating and shuffling players with roles, genes, and suspicion scores.
    
    Args:
        genes_list: List of 5 numpy arrays where:
                   - 2 spies have gene arrays of size 10
                   - 3 resistance members have gene arrays of size 14
        
    Returns:
        List of 5 shuffled Player objects with:
        - IDs (0-4)
        - Roles (0 = resistance, 1 = spy)
        - Genes (spies: length 10, resistance: length 14)
        - Suspicion scores initialized (resistance: self-score = -1000)
    """
    # Validate input
    if len(genes_list) != 5:
        raise ValueError("Expected 5 gene arrays (2 spies len=10, 3 resistance len=14)")
    
    # Separate genes by role
    spy_genes = [g for g in genes_list if len(g) == 10]
    resistance_genes = [g for g in genes_list if len(g) == 14]
    
    if len(spy_genes) != 2 or len(resistance_genes) != 3:
        raise ValueError("Incorrect gene distribution: Expected 2xlen10 (spies) and 3xlen14 (resistance)")
    
    # Create players (temporarily without IDs)
    players = [
        *[Player(id=None, role=1, genes=genes) for genes in spy_genes],     # Spies (role=1)
        *[Player(id=None, role=0, genes=genes) for genes in resistance_genes]  # Resistance (role=0)
    ]
    
    # Shuffle and assign IDs
    random.shuffle(players)
    for player_id, player in enumerate(players):
        player.id = player_id
        if player.role == 0:  # Initialize resistance suspicion scores
            player.suspicion_scores = np.zeros(5, dtype=np.float32)
            player.suspicion_scores[player_id] = -1000.0  # Self-marking
    
    return players

def simulate_game(list_of_players: List[Player], verbose=False) -> str:
    mission_sizes = [2, 3, 2, 3, 3]
    players = setup_game(list_of_players)
    completed_missions = 0
    sabotaged_missions = 0
    mission_leader_idx = 0  # Track leader by index instead of object
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
            team = propose_team(players[mission_leader_idx], players, team_size, mission_num)
            if verbose: 
              print(f"Mission Leader: Player {players[mission_leader_idx].id}")
              print(f"Proposed Team: {[str(p.id) + ': ' + ('Resistance' if p.role == 0 else 'Spy') for p in team]}")
            
            # Voting
            vote_results = {}  # {player_id: int}
            for p in players:
              vote_results[p.id] = spy_vote(p, 
                                            mission_leader_idx,
                                            team, 
                                            mission_num, 
                                            completed_missions, 
                                            sabotaged_missions, 
                                            voting_round
                                            ) if p in spies else resistance_vote(
                                                 p,
                                                 team,
                                                 mission_leader_idx,
                                                 mission_num, 
                                                 voting_round, 
                                                 players
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
        sabotage_count = sum(1 for p in team if p in spies and sabotage(p, team, mission_num, completed_missions, sabotaged_missions))
        if sabotage_count >= 1:
            if verbose: print(f"Number of sabotages: {sabotage_count}")
            sabotaged_missions += 1
            if verbose: print("Mission Failed! (Sabotaged)")
        else:
            completed_missions += 1
            mission_successful = True
            if verbose: print("Mission Succeeded!")

        if verbose: print(f"Scores so far: sabotaged missions = {sabotaged_missions} / completed missions: {completed_missions}")


        # Win Check
        if completed_missions >= 3:
            if verbose: print("Resistance wins - 3 successful missions!")
            return 'resistance'
        elif sabotaged_missions >= 3:
            if verbose: print("Spies win - 3 sabotaged missions!")
            return 'spies'

        for p in resistance:
           update_suspicion(p,
                            players,
                            team,
                            mission_leader_idx, 
                            team_size, 
                            sabotage_count,
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

spy_pop = np.random.uniform(low=0.0, high=1.0, size=(100, 10))
resistance_pop = np.random.uniform(low=-1.0, high=1.0, size=(100, 14))

"""
win = 0
n_games = 100_000

for i in range(n_games):
  r = random.sample(list(resistance_pop), k=3)
  s = random.sample(list(spy_pop), k=2)
  players = r + s
  winner = simulate_game(players, verbose=False)
  if winner == 'resistance':
    win += 1

print(f"Resistance win rate: {win/n_games:.3f}")
"""